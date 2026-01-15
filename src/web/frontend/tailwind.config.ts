import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Custom palette from requirements
        background: '#0B0F14',
        surface: '#111827',
        'surface-2': '#0F172A',
        primary: '#D4AF37',
        'primary-2': '#C8A951',
        text: '#E5E7EB',
        muted: '#9CA3AF',
        border: '#1F2937',
        success: '#22C55E',
        warning: '#F59E0B',
        danger: '#EF4444',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
export default config
