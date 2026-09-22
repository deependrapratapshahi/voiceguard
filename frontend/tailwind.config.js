/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0f14",
        panel: "#111826",
        panelBorder: "#1f2937",
        accent: "#22d3ee",
        risklow: "#22c55e",
        riskmed: "#eab308",
        riskhigh: "#f97316",
        riskcrit: "#ef4444"
      }
    }
  },
  plugins: []
}
