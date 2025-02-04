st.markdown(f"Enter text for analysis (Maximum: {MAX_WORDS} words):", unsafe_allow_html=True)
   text_input = st.text_area("Enter text for analysis:", height=150,  key="text_area",  max_chars= MAX_WORDS * 7) # this assumes that every text is around has 7 character including space
   word_count = len(text_input.split())
   st.markdown(f'<div id="word-count">Word count: {word_count}</div>', unsafe_allow_html=True)  # show words limit