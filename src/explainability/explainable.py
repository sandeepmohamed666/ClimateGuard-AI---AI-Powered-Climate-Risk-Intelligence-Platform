"""
Explainability adapter that exposes a unified `ModelExplainer` interface.

This module will prefer the SHAP-based explainer (if SHAP is importable)
and fall back to the sklearn-native explainer. It normalizes method names
and attributes so notebooks and other code can interact with a single
`ModelExplainer` implementation.
"""
from typing import Optional
import logging
import joblib

logger = logging.getLogger(__name__)

try:
    # Prefer SHAP explainer when available
    from .shap_explainer import ModelExplainer as SHAPModelExplainer  # type: ignore
    _HAS_SHAP_IMPL = True
except Exception:
    SHAPModelExplainer = None
    _HAS_SHAP_IMPL = False

try:
    from .model_explainer import ModelExplainer as SKLearnModelExplainer  # type: ignore
    _HAS_SKLEARN_IMPL = True
except Exception:
    SKLearnModelExplainer = None
    _HAS_SKLEARN_IMPL = False


class ModelExplainer:
    """Adapter that normalizes the explainer interface across implementations.

    Args:
        model: trained model
        data: DataFrame used for explainer (train or background data)
        feature_names: optional list of feature names
    """

    def __init__(self, model, data, feature_names: Optional[list] = None):
        self.model = model
        self.X = data
        self.feature_names = feature_names or (list(data.columns) if data is not None else None)
        self.preprocessed = False

        # Select implementation. Prefer SHAP for tree-based models; prefer sklearn explainer for linear/non-tree models.
        prefer_sklearn = False
        # heuristic: models with feature_importances_ or from sklearn.ensemble are tree-based
        if hasattr(model, 'feature_importances_') or model.__class__.__module__.startswith('sklearn.ensemble'):
            prefer_sklearn = False
        else:
            # for linear models or others, prefer sklearn explainer when available
            prefer_sklearn = True

        # align provided data to model's expected features when possible
        aligned_data = data
        model_feat_names = getattr(model, 'feature_names_in_', None)
        expected = getattr(model, 'n_features_in_', None)
        if aligned_data is not None:
            try:
                if model_feat_names is not None:
                    # select only columns the model expects (if present)
                    aligned_cols = [c for c in model_feat_names if c in aligned_data.columns]
                    aligned_data = aligned_data[aligned_cols].copy()
                elif expected is not None and aligned_data.shape[1] != expected:
                    # fallback: use first `expected` numeric columns
                    aligned_data = aligned_data.select_dtypes(include=['number']).iloc[:, :expected].copy()
            except Exception:
                logger.exception('Failed to align data to model features; proceeding with original data')

        aligned_feature_names = (aligned_data.columns.tolist() if (aligned_data is not None and hasattr(aligned_data, 'columns')) else self.feature_names)

        if _HAS_SHAP_IMPL and not prefer_sklearn:
            logger.info("Using SHAP-based ModelExplainer")
            self._impl = SHAPModelExplainer(model, aligned_data, feature_names=aligned_feature_names)
        elif _HAS_SKLEARN_IMPL:
            logger.info("Using sklearn-based ModelExplainer (fallback)")
            self._impl = SKLearnModelExplainer(model, aligned_data, feature_names=aligned_feature_names)
        else:
            raise ImportError("No explainer implementation available (neither SHAP nor sklearn explainer found)")

        # Mirror commonly-accessed attributes expected by notebooks
        self.explainer = getattr(self._impl, 'explainer', None)
        self.scaler = getattr(self._impl, 'scaler', None)
        self.shap_values = getattr(self._impl, 'shap_values', None)

    def preprocess_data(self):
        if hasattr(self._impl, 'preprocess_data'):
            self._impl.preprocess_data()
        self.preprocessed = True
        # keep mirrored attributes
        self.scaler = getattr(self._impl, 'scaler', None)

    def create_tree_explainer(self):
        # Try to create a tree explainer; on failure, attempt linear explainer,
        # and finally fall back to the sklearn-based explainer implementation.
        try:
            if hasattr(self._impl, 'create_tree_explainer'):
                self._impl.create_tree_explainer()
            self.explainer = getattr(self._impl, 'explainer', None)
        except Exception:
            logger.exception('TreeExplainer failed; attempting LinearExplainer')
            try:
                # Try to create linear explainer via adapter helper
                self.create_linear_explainer()
                self.explainer = getattr(self._impl, 'explainer', None)
            except Exception:
                logger.exception('LinearExplainer creation failed')
                # Fallback: use sklearn explainer implementation if available
                if _HAS_SKLEARN_IMPL and not isinstance(self._impl, SKLearnModelExplainer):
                    logger.info('Falling back to sklearn-based explainer implementation')
                    self._impl = SKLearnModelExplainer(self.model, self.X, feature_names=self.feature_names)
                    self.explainer = getattr(self._impl, 'explainer', None)

        # If no explainer instance was created by SHAP, ensure we fall back to sklearn explainer
        if getattr(self, 'explainer', None) is None and _HAS_SKLEARN_IMPL and not isinstance(self._impl, SKLearnModelExplainer):
            logger.info('No SHAP explainer available; switching to sklearn explainer fallback')
            self._impl = SKLearnModelExplainer(self.model, self.X, feature_names=self.feature_names)
            self.explainer = getattr(self._impl, 'explainer', None)

    def create_kernel_explainer(self):
        if hasattr(self._impl, 'create_kernel_explainer'):
            self._impl.create_kernel_explainer()
        self.explainer = getattr(self._impl, 'explainer', None)

    def create_linear_explainer(self):
        # Some legacy notebooks expect a create_linear_explainer helper.
        # Try to construct one from SHAP if available, otherwise no-op.
        try:
            import shap  # type: ignore
        except Exception:
            return
        if hasattr(shap, 'LinearExplainer') and hasattr(self.model, '__class__'):
            try:
                # Build a background DataFrame aligned to the model's expected features
                background_df = None
                if self.X is not None:
                    # prefer numeric columns and fillna
                    background_df = self.X.select_dtypes(include=['number']).fillna(0).copy()

                # If model exposes feature names or expected feature count, align
                expected = getattr(self.model, 'n_features_in_', None)
                model_feat_names = getattr(self.model, 'feature_names_in_', None)

                if model_feat_names is not None and background_df is not None:
                    # select only the features the model expects if present in background
                    available = [f for f in model_feat_names if f in background_df.columns]
                    if len(available) == len(model_feat_names):
                        background_df = background_df[model_feat_names]
                    else:
                        # fallback to intersecting columns in original order
                        background_df = background_df[[c for c in model_feat_names if c in background_df.columns]]

                if expected is not None and background_df is not None and background_df.shape[1] != expected:
                    # try to reduce or pad columns to match expected count
                    if background_df.shape[1] > expected:
                        background_df = background_df.iloc[:, :expected]
                    else:
                        # cannot pad sensibly; raise to be handled below
                        raise ValueError(f"Background data has {background_df.shape[1]} features but model expects {expected}")

                background_values = None
                if background_df is not None:
                    background_values = background_df.values[:100]

                self._impl.explainer = shap.LinearExplainer(self.model, background_values, feature_names=(background_df.columns.tolist() if background_df is not None else self.feature_names))
                self.explainer = self._impl.explainer
            except Exception:
                logger.exception('Failed to create LinearExplainer')
                raise

    def calculate_shap_values(self, X_test=None, max_samples: int = 100):
        # Delegate to underlying impl, adapting argument name differences if needed
        if X_test is None and X_test != getattr(self._impl, 'X_train', None):
            X_test = self.X

        # Many implementations accept X_test kw; try to call with that then fallback
        if hasattr(self._impl, 'calculate_shap_values'):
            try:
                vals = self._impl.calculate_shap_values(X_test=X_test, max_samples=max_samples)
            except TypeError:
                vals = self._impl.calculate_shap_values(X_test, max_samples)
            self.shap_values = getattr(self._impl, 'shap_values', vals)
            return self.shap_values
        else:
            # Fallback: create zero SHAP values placeholder
            import numpy as _np
            if X_test is None:
                if self.X is None:
                    raise ValueError("No data available to synthesize SHAP placeholders")
                n_rows = min(max_samples, len(self.X))
                n_feats = len(self.feature_names) if self.feature_names is not None else (self.X.shape[1])
            else:
                n_rows = min(max_samples, len(X_test))
                n_feats = X_test.shape[1]

            vals = _np.zeros((n_rows, n_feats))
            self.shap_values = vals
            return self.shap_values

    def get_feature_importance(self):
        return self._impl.get_feature_importance()

    def save(self, filepath: str):
        """Save the explainer adapter or its implementation to a pickle file."""
        try:
            if hasattr(self._impl, 'save'):
                return self._impl.save(filepath)
            joblib.dump(self, filepath)
            logger.info(f"Explainer adapter saved to {filepath}")
            return filepath
        except Exception:
            logger.exception(f"Failed to save explainer adapter to {filepath}; saving implementation only")
            joblib.dump(self._impl, filepath)
            logger.info(f"Explainer implementation saved to {filepath}")
            return filepath

    def plot_summary_plot(self, save_path: Optional[str] = None):
        # Prefer SHAP summary_plot, fallback to model feature importance plot
        if hasattr(self._impl, 'plot_summary_plot'):
            return self._impl.plot_summary_plot(save_path=save_path) if save_path is not None else self._impl.plot_summary_plot()
        if hasattr(self._impl, 'plot_feature_importance'):
            return self._impl.plot_feature_importance(save_path=save_path)
        # No-op fallback
        logger.warning('No summary or feature importance plot method available on impl')
        return None

    def plot_bar_plot(self, save_path: Optional[str] = None):
        # Prefer SHAP bar plot, fallback to permutation or feature importance
        if hasattr(self._impl, 'plot_bar_plot'):
            return self._impl.plot_bar_plot(save_path=save_path) if save_path is not None else self._impl.plot_bar_plot()
        if hasattr(self._impl, 'plot_permutation_importance'):
            return self._impl.plot_permutation_importance(save_path=save_path)
        if hasattr(self._impl, 'plot_feature_importance'):
            return self._impl.plot_feature_importance(save_path=save_path)
        logger.warning('No bar/permutation/feature importance plot method available on impl')
        return None

    def plot_waterfall_plot(self, instance_idx: int = 0, save_path: Optional[str] = None):
        try:
            fn = getattr(self._impl, 'plot_waterfall_plot', None)
        except Exception:
            fn = None
        if callable(fn):
            return fn(instance_idx=instance_idx, save_path=save_path) if save_path is not None else fn(instance_idx=instance_idx)
        logger.warning('Waterfall plot not available for this explainer implementation')
        return None

    def plot_decision_plot(self, instance_idx: int = 0, save_path: Optional[str] = None):
        try:
            fn = getattr(self._impl, 'plot_decision_plot', None)
        except Exception:
            fn = None
        if callable(fn):
            return fn(instance_idx=instance_idx, save_path=save_path) if save_path is not None else fn(instance_idx=instance_idx)
        logger.warning('Decision plot not available for this explainer implementation')
        return None

    def explain_instance(self, sample):
        # Accept DataFrame or dict/Series; ensure DataFrame
        import pandas as pd
        if not isinstance(sample, pd.DataFrame):
            sample = pd.DataFrame(sample)
        raw = self._impl.explain_instance(sample)

        # Normalize to a common structure expected by the notebook
        # target keys: base_value, prediction, feature_contributions (feat -> {shap_value, impact})
        out = {}
        out['base_value'] = float(raw.get('base_value', 0.0)) if isinstance(raw, dict) else 0.0
        out['prediction'] = float(raw.get('prediction', raw.get('prediction', raw.get('pred', float('nan'))))) if isinstance(raw, dict) else float('nan')

        contributions = {}
        raw_contrib = raw.get('feature_contributions', {}) if isinstance(raw, dict) else {}
        for feat, val in raw_contrib.items():
            # val might have 'shap_value' or 'importance'
            if isinstance(val, dict):
                shap_v = val.get('shap_value', val.get('importance', 0.0))
                impact = val.get('impact', 'neutral')
            else:
                shap_v = float(val)
                impact = 'neutral'
            try:
                shap_v = float(shap_v)
            except Exception:
                shap_v = 0.0
            contributions[feat] = {'shap_value': shap_v, 'impact': impact}

        out['feature_contributions'] = contributions
        return out

    def generate_explanation_report(self, X_test=None):
        if X_test is None:
            X_test = self.X

        try:
            return self._impl.generate_explanation_report(X_test)
        except TypeError:
            # Fallback: call without args if the implementation doesn't accept X_test
            return self._impl.generate_explanation_report()

    def get_explanation_report(self):
        return self._impl.get_explanation_report()

    def print_report(self):
        return self._impl.print_report()

    def get_impl(self):
        """Return the underlying implementation (for advanced usage)."""
        return self._impl
