export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "#faf9f7",
        surface: "#faf9f7",
        "surface-container": "#eeeeeb",
        "surface-container-low": "#f4f4f1",
        "surface-container-lowest": "#ffffff",
        "surface-container-high": "#e8e8e6",
        primary: "#173422",
        "primary-container": "#2e4b37",
        "on-primary": "#ffffff",
        "on-surface": "#1a1c1b",
        "on-surface-variant": "#424843",
        secondary: "#556157",
        "secondary-container": "#d8e6d9",
        "on-secondary-container": "#5b685d",
        tertiary: "#183145",
        outline: "#727972",
        "outline-variant": "#c2c8c0",
        error: "#ba1a1a",
      },
      borderRadius: {
        DEFAULT: "0.25rem",
        lg: "0.5rem",
        xl: "0.75rem",
        full: "9999px",
      },
      fontFamily: {
        body: ["Inter", "sans-serif"],
        display: ["Plus Jakarta Sans", "sans-serif"],
      },
    },
  },
  plugins: [],
};
