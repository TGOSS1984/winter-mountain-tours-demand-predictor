mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
base = 'dark'\n\
primaryColor = '#F97316'\n\
backgroundColor = '#020617'\n\
secondaryBackgroundColor = '#0f172a'\n\
textColor = '#e5e7eb'\n\
font = 'sans serif'\n\
\n\
" > ~/.streamlit/config.toml

