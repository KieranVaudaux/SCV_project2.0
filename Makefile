st:

	cd notebooks
	@streamlit run streamlit_app.py --theme.primaryColor="#2c71de" --theme.backgroundColor="#678fd2" --theme.secondaryBackgroundColor="#767a96" --theme.textColor="#dfe4ea"
  
requirements:
		
		$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
 
