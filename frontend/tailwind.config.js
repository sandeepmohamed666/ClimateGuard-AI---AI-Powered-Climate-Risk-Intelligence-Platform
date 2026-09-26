module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#eef8ff',
          100: '#d8efff',
          200: '#b4ddff',
          300: '#83c7ff',
          400: '#4ea8ff',
          500: '#1d86ff',
          600: '#0a65e8',
          700: '#0c4fb2',
          800: '#0f3d84',
          900: '#11305f'
        }
      },
      boxShadow: {
        glass: '0 20px 80px rgba(15, 23, 42, 0.08)',
      }
    }
  },
  plugins: [],
};
