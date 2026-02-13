/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        alfred: {
          bg: '#0a0a0a',
          surface: '#1a1a1a',
          border: '#2f2f2f',
          hover: '#333333',
          input: '#1e1e1e',
          text: '#e0e0e0',
          muted: '#888888',
          accent: '#f97316',  // orange-500
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['SF Mono', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
