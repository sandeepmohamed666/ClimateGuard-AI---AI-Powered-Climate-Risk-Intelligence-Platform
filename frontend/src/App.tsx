import { Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Weather from './pages/Weather';
import Forecasting from './pages/Forecasting';
import RiskAnalysis from './pages/RiskAnalysis';
import ExplainableAI from './pages/ExplainableAI';
import FeatureEngineering from './pages/FeatureEngineering';
import DataExplorer from './pages/DataExplorer';
import ModelPerformance from './pages/ModelPerformance';
import Settings from './pages/Settings';
import About from './pages/About';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';

function App() {
  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.15),_transparent_38%),radial-gradient(circle_at_bottom_right,_rgba(16,185,129,0.12),_transparent_32%),#020617]">
      <div className="mx-auto flex min-h-screen max-w-[1600px] flex-col lg:flex-row">
        <Sidebar />
        <div className="flex-1 px-4 py-4 lg:px-8">
          <Header />
          <Routes>
            <Route path="/" element={<Navigate replace to="/dashboard" />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/weather" element={<Weather />} />
            <Route path="/forecasting" element={<Forecasting />} />
            <Route path="/risk" element={<RiskAnalysis />} />
            <Route path="/explainable-ai" element={<ExplainableAI />} />
            <Route path="/feature-engineering" element={<FeatureEngineering />} />
            <Route path="/data-explorer" element={<DataExplorer />} />
            <Route path="/model-performance" element={<ModelPerformance />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}

export default App;
