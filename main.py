import os
import streamlit as st
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import pandas as pd
from PIL import Image
import easyocr as ocr
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_1samp, ttest_rel, ttest_ind, spearmanr, chisquare
from scipy.stats import f_oneway, pearsonr, chi2_contingency
from statsmodels.tsa.stattools import acf
import seaborn as sns
from statsmodels.stats.weightstats import ztest
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr

load_dotenv()  # take environment variables from .env (especially openai api key)

file_path = "VE_Storage_Database.pkl"

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.4)

# Function to handle speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.write("Listening...")
        audio = recognizer.listen(source)

    try:
        st.sidebar.write("Processing...")
        query = recognizer.recognize_google(audio, language="en-US")  # Adjust language as needed
        return query
    except sr.UnknownValueError:
        st.sidebar.write("Sorry, I could not understand your speech.")
        return ""
    except sr.RequestError:
        st.sidebar.write("Sorry, there was an error processing your request.")
        return ""

def answer_general_gpt_question(question):
    st.header("General GPT Question")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Use the llm model to generate a response to the question
            chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            # result = chain({"question": query}, return_only_outputs=True)
            inputs = {"query": query}
            start_time = time.time()  # Record the start time
            result = chain(inputs, return_only_outputs=True)
            end_time = time.time()  # Record the end time

            elapsed_time = end_time - start_time  # Calculate elapsed time

            st.write("Answer:", result)
            st.write(f"Time taken to answer: {elapsed_time:.2f} seconds")

            # Store the question and its answer in the session state
            if "question_answers" not in st.session_state:
                st.session_state["question_answers"] = {}

            st.session_state["question_answers"][question] = {
                "answer": result,
                "time_taken": elapsed_time
            }

            # Display retained questions and answers
            st.header("Retained Questions and Answers")
            if "question_answers" in st.session_state and st.session_state["question_answers"]:
                for question, data in st.session_state["question_answers"].items():
                    if isinstance(data, dict):
                        st.write(f"Question: {question}")
                        st.write(f"Answer: {data.get('answer', 'Answer not available')}")
                        st.write(f"Time taken: {data.get('time_taken', 0):.2f} seconds")
                    else:
                        st.write(f"Question: {question}")
                        st.write("Answer: Data structure not recognized")

st.title("AnalytiBot: A Mini Customized Research Tool")

main_placeholder = st.empty()

st.sidebar.title("AB - Research Made Easy!!!")
voice_input_option = st.sidebar.checkbox("Use Voice Input")
# Sidebar section for general GPT questions
st.sidebar.header("GENERAL GPT BOT")
general_gpt_option = st.sidebar.checkbox("Answer General GPT Question")
if general_gpt_option:
    if voice_input_option:
        query = recognize_speech()
        st.sidebar.write("Voice input recognized:", query)
        answer_general_gpt_question(query)
    else:
        query = main_placeholder.text_input("Enter you QUESTION: ")
        if st.sidebar.button("Ask"):
            answer_general_gpt_question(query)


st.sidebar.header("DATA INPUT GPT BOT")

# Allow user to choose between URL or PDF file
data_input_choice = st.sidebar.radio("Choose Data Input Method:", ("CSV File", "Image", "PDF File", "URL Links"))

if data_input_choice == "URL Links":
    num_urls = st.sidebar.number_input("Number of URLs to Upload", min_value=1, max_value=10, value=3)
    urls = []
    for i in range(num_urls):
        url = st.sidebar.text_input(f"URL {i + 1}")
        urls.append(url)

    process_data_clicked = st.sidebar.button("Run URLs")

elif data_input_choice == "CSV File":
    uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type="csv")
    if uploaded_csv:
        st.header("Regression Analysis")
        regression_type = st.selectbox("Select Regression Type:", ("Linear Regression", "Multiple Regression"))

        if regression_type == "Linear Regression":
            st.write("Select dependent and independent variables:")
            data = pd.read_csv(uploaded_csv)
            columns = data.select_dtypes(include='number').columns.tolist()  # Select only numeric columns
            dependent_variable = st.selectbox("Select Dependent Variable:", columns)
            independent_variable = st.selectbox("Select Independent Variable:", columns)

        elif regression_type == "Multiple Regression":
            st.write("Select dependent and independent variables:")
            data = pd.read_csv(uploaded_csv)
            columns = data.select_dtypes(include='number').columns.tolist()  # Select only numeric columns
            dependent_variable = st.selectbox("Select Dependent Variable:", columns)
            independent_variables = st.multiselect("Select Independent Variables:", columns)

        all_columns = data.columns.tolist()

        # Add inputs for statistical tests
        st.header("Statistical Tests")
        significance_level = st.number_input("Enter Significance Level (e.g., 0.05):", value=0.05, step=0.01)
        test_type = st.selectbox("Select Test Type:", (
            "t-Tests (One-sample)", "t-Tests (Independent samples)", "t-Tests (Paired samples)",
            "Analysis of Variance (One-way)", "Analysis of Variance (Two-way)", "Z-test",
            "Correlation Analysis", "Chi-square Test",
            "Moving Average", "Exponential Smoothing", "Autocorrelation", "Seasonality Analysis",
            "Histogram", "Box Plot", "Q-Q Plot",
            "Pearson Correlation Coefficient", "Spearman's Rank Correlation Coefficient",
            "Chi-square Test of Independence"))

        if test_type in ("t-Tests (One-sample)", "t-Tests (Independent samples)", "t-Tests (Paired samples)"):
            st.subheader("T-Test Variables")
            dependent_variable_ttest = st.selectbox("Select Dependent Variable for T-Test:", columns, key="ttest_dependent_variable")
            independent_variable_ttest = st.selectbox("Select Independent Variable for T-Test:", columns, key="ttest_independent_variable")
            equal_variances = st.checkbox("Equal Variances?")


        elif test_type in ("Analysis of Variance (One-way)", "Analysis of Variance (Two-way)"):
            st.subheader("ANOVA Variables")
            dependent_variable_anova = st.selectbox("Select Dependent Variable for ANOVA:", columns,
                                                    key="anova_dependent_variable")
            # One-way ANOVA selections (if applicable)
            if test_type == "Analysis of Variance (One-way)":
                independent_variable_anova = st.selectbox("Select Independent Variable for ANOVA:", all_columns,
                                                          key="anova_independent_variable")
            # Two-way ANOVA selections (if applicable)
            elif test_type == "Analysis of Variance (Two-way)":
                independent_variable1_anova = st.selectbox("Select 1st Independent Variable for ANOVA:", all_columns,
                                                           key="anova_independent_variable1")
                independent_variable2_anova = st.selectbox("Select 2nd Independent Variable for ANOVA:",
                                                           all_columns,
                                                           key="anova_independent_variable2")  # Ensure unique variables

        elif test_type == "Z-test":
            st.subheader("Z-test")
            variable1_ztest = st.selectbox("Select Variable 1 for Z-test:", columns, key="ztest_variable1")
            variable2_ztest = st.selectbox("Select Variable 2 for Z-test:", columns, key="ztest_variable2")

        elif test_type == "Correlation Analysis":
            st.subheader("Correlation Analysis")
            selected_variables_corr = st.multiselect("Select Variables for Correlation Analysis:", columns)

        elif test_type == "Chi-square Test":
            st.subheader("Chi-square Test")
            # Allow selecting multiple variables
            selected_variables = st.multiselect("Select Variables for Chi-square Test:", all_columns)


        elif test_type == "Moving Average":
            st.subheader("Moving Average")
            variable1_Movi_Avg = st.selectbox("Select Variable for Performing Moving Avg:", columns,
                                              key="Movi_Avg_variable1")
            window_size_ma = st.number_input("Enter Window Size for Moving Average:", min_value=1, max_value=len(data),
                                             value=10)
        elif test_type == "Exponential Smoothing":
            st.subheader("Exponential Smoothing")
            variable1_expo_smooth = st.selectbox("Select Variable for Performing Moving Avg:", columns,
                                              key="Expo_Smooth_variable1")
            smoothing_level = st.number_input("Enter Smoothing Level for Exponential Smoothing:", min_value=0.01,
                                              max_value=1.0, value=0.5)
        elif test_type == "Autocorrelation":
            st.subheader("Autocorrelation")
            selected_variables_acf = st.multiselect("Select Variable(s) for Autocorrelation:", columns)
            lag_acf = st.number_input("Enter Lag for Autocorrelation:", min_value=0, max_value=len(data) - 1, value=1)

        elif test_type == "Seasonality Analysis":
            st.subheader("Seasonality Analysis")
            time_series_data = st.file_uploader("Upload Time Series Data (CSV file)", type="csv")
            seasonality_frequency = st.selectbox("Select Seasonality Frequency:", ["Daily", "Weekly", "Monthly"])
            visualization_method = st.selectbox("Select Visualization Method:", ["Line Plot", "Seasonal Decomposition"])

        elif test_type == "Histogram":
            st.subheader("Histogram")
            variable_for_histogram = st.selectbox("Select Variable for Histogram:", columns)
            num_bins = st.number_input("Number of Bins:", min_value=1, value=10)

        elif test_type == "Box Plot":
            st.subheader("Box Plot")
            variable_for_box_plot = st.selectbox("Select Variable for Box Plot:", columns)
            group_by_variable = st.selectbox("Group Data by Variable (Optional):", [None] + columns)

        elif test_type == "Q-Q Plot":
            st.subheader("Q-Q Plot")
            variable_for_qq_plot = st.selectbox("Select Variable for Q-Q Plot:", columns)
            distribution_to_compare = st.selectbox("Distribution to Compare Against:", ["Normal", "Uniform"])


        elif test_type == "Pearson Correlation Coefficient":
            st.subheader("Pearson Correlation Coefficient")
            variable1_pearson = st.selectbox("Select Variable 1 for Pearson Correlation Coefficient:", columns,
                                             key="pearson_variable1")
            variable2_pearson = st.selectbox("Select Variable 2 for Pearson Correlation Coefficient:", columns,
                                             key="pearson_variable2")

        elif test_type == "Spearman's Rank Correlation Coefficient":
            st.subheader("Spearman's Rank Correlation Coefficient")
            variable1_spearman = st.selectbox("Select Variable 1 for Spearman's Rank Correlation Coefficient:", columns,
                                              key="spearman_coeff_variable1")
            variable2_spearman = st.selectbox("Select Variable 2 for Spearman's Rank Correlation Coefficient:", columns,
                                              key="spearman_coeff_variable2")

        elif test_type == "Chi-square Test of Independence":
            st.subheader("Chi-square Test of Independence")
            variable1_chi_square = st.selectbox("Select Variable 1 for Chi-square Test of Independence:", all_columns,
                                                key="chi_square_independence_variable1")
            variable2_chi_square = st.selectbox("Select Variable 2 for Chi-square Test of Independence:", all_columns,
                                                key="chi_square_independence_variable2")

    process_data_clicked = st.sidebar.button("Run CSV")

elif data_input_choice == "PDF File":
    uploaded_file = st.sidebar.file_uploader("Upload PDF File", type="pdf")

    process_data_clicked = st.sidebar.button("Run PDF")

elif data_input_choice == "Image":
    image = st.sidebar.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'])

    process_data_clicked = st.sidebar.button("Run Image")


if data_input_choice in ("PDF File", "URL Links", "Image"):
    # Use session_state to store previous prompts and questions
    if "question_answers" not in st.session_state:
        st.session_state["question_answers"] = {}

    if "question_timings" not in st.session_state:
        st.session_state["question_timings"] = {}

    if process_data_clicked:
        if data_input_choice == "URL Links":
            # load data from URLs
            loader = UnstructuredURLLoader(urls=urls)
            main_placeholder.text("Loading...")
            data = loader.load()

            # split data
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            main_placeholder.text("Splitting the text..")
            docs = text_splitter.split_documents(data)

            # create embeddings and save it to FAISS index
            embeddings = OpenAIEmbeddings()
            vectorstore_openai = FAISS.from_documents(docs, embeddings)
            main_placeholder.text("Working on VE...")
            time.sleep(2)

            # Save the FAISS index to a pickle file
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

        elif data_input_choice == "PDF File":
            # load data from PDF file
            if uploaded_file:
                bytes_data = uploaded_file.read()
                with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
                    tmp.write(bytes_data)  # write data from the uploaded file into it
                    data = PyPDFLoader(tmp.name).load()  # <---- now it works!
                os.remove(tmp.name)

                # split data
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                main_placeholder.text("Splitting the text..")
                docs = text_splitter.split_documents(data)

                # create embeddings and save it to FAISS index
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                main_placeholder.text("Working on VE...")
                time.sleep(2)

                # Save the FAISS index to a pickle file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)

        elif data_input_choice == "Image":
            # image uploader
            # image = st.sidebar.file_uploader(label="Upload your image here", type=['png', 'jpg', 'jpeg'])
            @st.cache_data
            def load_model():
                reader = ocr.Reader(['en'], model_storage_directory='.')
                return reader


            reader = load_model()  # load model
            if image is not None:
                input_image = Image.open(image)  # read image
                st.image(input_image)  # display image
                with st.spinner("🤖 AB is at Work! "):
                    result = reader.readtext(np.array(input_image))
                    result_text = "\n".join([text[1] for text in result])  # Concatenate each line of text with newline
                    # Display the entire text as a single paragraph
                    st.write(result_text)

                    # Create PDF from the extracted text using reportlab
                    pdf_file_path = "image_text.pdf"  # Output PDF file path
                    c = canvas.Canvas(pdf_file_path, pagesize=letter)
                    # Write text to PDF with proper formatting
                    c.setFont("Helvetica", 12)
                    text_lines = result_text.split("\n")
                    y_coordinate = 750
                    for line in text_lines:
                        c.drawString(100, y_coordinate, line)
                        y_coordinate -= 15  # Adjust line spacing as needed
                    c.save()

                    # Provide the PDF file as output
                    with open(pdf_file_path, "rb") as pdf_file:
                        pdf_bytes = pdf_file.read()
                        st.download_button("Download PDF", pdf_bytes, file_name="image_text.pdf",
                                           mime="application/pdf")

                    # Debug statement
                    print("PDF creation successful.")


                st.balloons()
            else:
                st.write("Upload an Image")

    if voice_input_option:
        query = recognize_speech()
        st.sidebar.write("Voice input recognized:", query)
    else:
        query = main_placeholder.text_input("Enter you QUESTION: ")

    if query:
        start_time = time.time()  # Record the start time
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                if data_input_choice == "Image":
                    # Use RetrievalQAChain for images
                    chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    # result = chain({"question": query}, return_only_outputs=True)
                    inputs = {"query": query}
                    result = chain(inputs, return_only_outputs=True)
                    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                    st.header("Answer")
                    st.write(result)
                else:
                    # Use RetrievalQAWithSourcesChain for URLs and PDFs
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                    result = chain({"question": query}, return_only_outputs=True)
                    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
                    st.header("Answer")
                    st.write(result["answer"])

                # Calculate the time taken to answer the question
                end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Time taken to answer: {elapsed_time:.2f} seconds")

                if data_input_choice == "Image":
                    # Store the question, its answer, and the time taken in the session_state
                    st.session_state["question_answers"][query] = result
                    st.session_state["question_timings"][query] = elapsed_time
                else:
                    # Store the question, its answer, and the time taken in the session_state
                    st.session_state["question_answers"][query] = result["answer"]
                    st.session_state["question_timings"][query] = elapsed_time

                # Display sources, if available (for URLs and PDFs)
                if data_input_choice in ["URLs", "PDF File"]:
                    sources = result.get("sources", "")
                    if sources:
                        st.subheader("Sources:")
                        sources_list = sources.split("\n")  # Split the sources by newline
                        for source in sources_list:
                            st.write(source)

    # Display all asked questions with their answers and timings
    if st.session_state["question_answers"]:
        st.header("Asked Questions with Answers")
        for question, answer in st.session_state["question_answers"].items():
            st.write(f"Question: {question}")
            st.write(f"Answer: {answer}")
            # Check if the question exists in question_timings before accessing its timing
            try:
                st.write(f"Time taken: {st.session_state['question_timings'][question]:.2f} seconds")
            except KeyError:
                st.write("Time taken: Not available")


if data_input_choice == "CSV File":
    # Descriptive Analysis and
    if process_data_clicked and data_input_choice == "CSV File":
        st.header("Descriptive Analysis")
        st.write("Summary Statistics:")
        st.write(data.describe())

        if regression_type == "Linear Regression" and dependent_variable and independent_variable:
            X = data[independent_variable].values.reshape(-1, 1)
            y = data[dependent_variable].values.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)

            st.write("Regression Results:")
            st.write(f"Intercept: {model.intercept_}")
            st.write(f"Coefficient: {model.coef_[0]}")

            # Plotting the best-fit line
            st.header("Regression Plot")
            plt.scatter(X, y, label='Actual data points')
            plt.plot(X, predictions, color='red', label='Best-fit line')
            plt.xlabel('Independent Variable')
            plt.ylabel('Dependent Variable')
            plt.title('Linear Regression Analysis')
            plt.legend()
            st.pyplot(plt)

        elif regression_type == "Multiple Regression" and dependent_variable and independent_variables:
            X = data[independent_variables]
            y = data[dependent_variable]
            X = sm.add_constant(X)  # Add constant for intercept
            model = sm.OLS(y, X).fit()
            predictions = model.predict(X)

            st.write("Regression Results:")
            st.write(model.summary())

            st.write("Regression Equation:")
            st.write(model.params)

            # Plotting the best-fit line
            st.header("Regression Plot")
            plt.scatter(y, predictions, label='Actual vs Predicted')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Multiple Regression Analysis')
            plt.legend()
            st.pyplot(plt)

        # Perform statistical tests
        elif test_type == "t-Tests (One-sample)":
            if dependent_variable_ttest and independent_variable_ttest:
                t_statistic, p_value = ttest_1samp(data[independent_variable_ttest],
                                                   data[dependent_variable_ttest].mean())
                st.write("T-Statistic:", t_statistic)
                st.write("P-Value:", p_value)
                st.write("Conclusion:",
                         "Reject Null Hypothesis" if p_value < significance_level else "Fail to Reject Null Hypothesis")

        elif test_type == "t-Tests (Independent samples)":
            if dependent_variable_ttest and independent_variable_ttest:
                t_statistic, p_value = ttest_ind(data[independent_variable_ttest], data[dependent_variable_ttest],
                                                 equal_var=equal_variances)
                st.write("T-Statistic:", t_statistic)
                st.write("P-Value:", p_value)
                st.write("Conclusion:",
                         "Reject Null Hypothesis" if p_value < significance_level else "Fail to Reject Null Hypothesis")

        elif test_type == "t-Tests (Paired samples)":
            if dependent_variable_ttest and independent_variable_ttest:
                t_statistic, p_value = ttest_rel(data[independent_variable_ttest], data[dependent_variable_ttest])
                st.write("T-Statistic:", t_statistic)
                st.write("P-Value:", p_value)
                st.write("Conclusion:",
                         "Reject Null Hypothesis" if p_value < significance_level else "Fail to Reject Null Hypothesis")

        if test_type == "Analysis of Variance (One-way)":

            if dependent_variable_anova and independent_variable_anova:
                model = ols(f'{dependent_variable_anova} ~ C({independent_variable_anova})',
                            data).fit()  # Use categorical encoding for one-way ANOVA
                anova_table = sm.stats.anova_lm(model, typ=1)
                test_result = f_oneway(data[independent_variable_anova], data[dependent_variable_anova])
                st.write("ANOVA Table (One-way):")
                st.write(anova_table)
                st.write("F-Statistic:", test_result[0])
                st.write("P-Value:", test_result[1])
                st.write("Conclusion:", "Reject Null Hypothesis" if test_result[
                                                                        1] < significance_level else "Fail to Reject Null Hypothesis")


        elif test_type == "Analysis of Variance (Two-way)":
            if dependent_variable_anova and independent_variable1_anova and independent_variable2_anova:
                model = ols(f'{dependent_variable_anova} ~ C({independent_variable1_anova}) + C({independent_variable2_anova})',data).fit()  # Use categorical encoding for two-way ANOVA
                anova_table = sm.stats.anova_lm(model, typ=3)  # typ=3 for two-way interaction
                test_result = f_oneway(data[dependent_variable_anova], data[independent_variable1_anova],
                                       data[independent_variable2_anova])  # Include all factors for two-way test
                st.write("ANOVA Table (Two-way):")
                st.write(anova_table)
                st.write("F-Statistic:", test_result[0])
                st.write("P-Value:", test_result[1])
                st.write("Conclusion:", "Reject Null Hypothesis" if test_result[
                                                                        1] < significance_level else "Fail to Reject Null Hypothesis")

        # Perform Z-test
        if test_type == "Z-test":
            if variable1_ztest and variable2_ztest:
                # Extract data for selected variables
                data1 = data[variable1_ztest]
                data2 = data[variable2_ztest]

                # Perform Z-test (assuming data1 and data2 are normally distributed)
                z_statistic, p_value = ztest(data1, data2)

                # Print Z-test results
                st.write("Z-Statistic:", z_statistic)
                st.write("P-Value:", p_value)

                # Interpret Z-test results
                if p_value < significance_level:
                    st.write("Conclusion: Reject Null Hypothesis")
                else:
                    st.write("Conclusion: Fail to Reject Null Hypothesis")

        elif test_type == "Correlation Analysis":
            if selected_variables_corr:
                # Compute correlation coefficients
                correlation_matrix = data[selected_variables_corr].corr()
                # Display correlation table
                st.write("Correlation Table:")
                st.write(correlation_matrix)
                # Generate correlation matrix plot
                st.write("Correlation Matrix Plot:")
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
                st.pyplot(plt)


        elif test_type == "Chi-square Test":
            if selected_variables:
                for variable in selected_variables:
                    # Get the observed frequencies for each category
                    observed_values = data[variable].value_counts().values

                    # Assuming a uniform distribution as the expected distribution (adjust if needed)
                    expected_values = np.array([len(data) / len(data[variable].unique())] * len(
                        data[variable].unique()))

                    # Perform Chi-square test
                    chi2_stat, p_value = chisquare(observed_values, expected_values)

                    #  Display results
                    st.write(f"Chi-square Test for {variable}:")
                    st.write(f"Chi-square Statistic: {chi2_stat}")
                    st.write(f"P-Value: {p_value}")
                    st.write("Conclusion:",
                             "Reject Null Hypothesis" if p_value < significance_level else "Fail to Reject Null Hypothesis")

        elif test_type == "Moving Average":
            if window_size_ma:
                data['Moving Average'] = data[variable1_Movi_Avg].rolling(window=window_size_ma).mean()
                st.write("Moving Average:")
                st.write(data)

        elif test_type == "Exponential Smoothing":
            if smoothing_level:
                data['Exponential Smoothing'] = data[variable1_expo_smooth].ewm(alpha=smoothing_level).mean()
                st.write("Exponential Smoothing:")
                st.write(data)

        elif test_type == "Seasonality Analysis":
            if time_series_data:
                time_series = pd.read_csv(time_series_data)
                if visualization_method == "Line Plot":
                    st.write("Line Plot:")
                    st.line_chart(time_series)
                elif visualization_method == "Seasonal Decomposition":
                    decomposition = seasonal_decompose(time_series, model='additive',
                                                       freq=seasonality_frequency.lower())
                    st.write("Seasonal Decomposition:")
                    st.write(decomposition.plot())
                    st.write(decomposition.seasonal)

        elif test_type == "Autocorrelation":
            if selected_variables_acf and lag_acf:
                st.write("Autocorrelation Results:")
                for variable in selected_variables_acf:
                    acf_result = acf(data[variable], nlags=lag_acf)
                    st.write(f"Autocorrelation for {variable}:")
                    st.write(acf_result)

                    # Plot autocorrelation function
                    st.write(f"Autocorrelation Plot for {variable}:")
                    plt.figure(figsize=(10, 6))
                    # Plotting the Autocorrelation plot.
                    plt.acorr(data[variable], maxlags=10)
                    plt.xlabel("Lag")
                    plt.ylabel("Autocorrelation")
                    plt.title(f"Autocorrelation Function for {variable}")
                    st.pyplot(plt)

        elif test_type == "Histogram":
            if variable_for_histogram:
                st.write(f"Histogram for {variable_for_histogram}:")
                plt.figure(figsize=(10, 6))
                plt.hist(data[variable_for_histogram], bins=num_bins)
                plt.xlabel(variable_for_histogram)
                plt.ylabel('Frequency')
                st.pyplot(plt)

        elif test_type == "Box Plot":
            if variable_for_box_plot:
                st.write("Box Plot:")
                plt.figure(figsize=(10, 6))
                if group_by_variable:
                    data.boxplot(column=variable_for_box_plot, by=group_by_variable)
                else:
                    data.boxplot(column=variable_for_box_plot)
                plt.ylabel(variable_for_box_plot)
                st.pyplot(plt)

        elif test_type == "Q-Q Plot":
            if variable_for_qq_plot:
                st.write("Q-Q Plot:")
                plt.figure(figsize=(10, 6))
                if distribution_to_compare == "Normal":
                    qqplot(data[variable_for_qq_plot], line='s')
                elif distribution_to_compare == "Uniform":
                    qqplot(data[variable_for_qq_plot], line='q')
                st.pyplot(plt)

        elif test_type == "Pearson Correlation Coefficient":
            if variable1_pearson and variable2_pearson:
                correlation_coefficient = pearsonr(data[variable1_pearson], data[variable2_pearson])[0]
                st.write("Pearson Correlation Coefficient:", correlation_coefficient)
                st.write("Conclusion:", "Reject Null Hypothesis" if abs(
                    correlation_coefficient) > significance_level else "Fail to Reject Null Hypothesis")

        elif test_type == "Spearman's Rank Correlation Coefficient":
            if variable1_spearman and variable2_spearman:
                correlation_coefficient = spearmanr(data[variable1_spearman], data[variable2_spearman])[0]
                st.write("Spearman's Rank Correlation Coefficient:", correlation_coefficient)
                st.write("Conclusion:", "Reject Null Hypothesis" if abs(
                    correlation_coefficient) > significance_level else "Fail to Reject Null Hypothesis")

        elif test_type == "Chi-square Test of Independence":
            if variable1_chi_square and variable2_chi_square:
                contingency_table = pd.crosstab(data[variable1_chi_square], data[variable2_chi_square])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                st.write("Chi-square Test of Independence:")
                st.write("Chi-square Statistic:", chi2)
                st.write("P-Value:", p)
                st.write("Degrees of Freedom:", dof)
                st.write("Expected Frequencies:")
                st.write(expected)
                st.write("Conclusion:",
                         "Reject Null Hypothesis" if p < significance_level else "Fail to Reject Null Hypothesis")


