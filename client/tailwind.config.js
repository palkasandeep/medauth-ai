/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        medcore: {
          50: "#eff6ff",
          100: "#dbeafe",
          600: "#0f4c81",
          700: "#0c3d68",
          800: "#092f50",
        },
      },
    },
  },
  plugins: [],
};
