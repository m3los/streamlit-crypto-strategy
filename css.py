from streamlit import markdown


# sidebar, main title, divider, form buttons
def apply_css():
    markdown("""
  <style>
    .st-emotion-cache-16txtl3 {
      padding-top: 1rem;
    }
    .st-emotion-cache-z5fcl4 {
      padding-top: 1rem;
    }
    hr {
      margin-top: 1em;
    }
    .st-emotion-cache-522quz {
      # background-color: #2FBB5B;
      border: 2px outset;
          border-top-color: gray;
          border-left-color: gray;
          border-right-color: black;
          border-bottom-color: black;
    }
  </style>
""", unsafe_allow_html=True)
