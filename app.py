
import ollama
import streamlit as st
from vectordb import index_data, vector_store

st.set_page_config(
    page_title="RAG Vietnamese QA Chatbot",
    layout="wide",
    initial_sidebar_state="expanded"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar.expander("Settings", expanded=False):

    llm_model = st.selectbox(
        "LLM Models",
        ["qwen3:1.7b", "deepseek-r1", "llama3.2:3b"],
        index=0
    )

    temperature = st.slider("Temperature", 0.0, 1.0, value=0.3)
    max_tokens = st.slider("Max tokens", 0, 1000, value=700)

    if st.sidebar.button("Clear chat history", key="clear_btn"):

        st.session_state.messages = []
        st.rerun()

st.title("RAG Files Chatbot")

uploaded_file = st.file_uploader(
    "Upload a file", type=["pdf", "png", "jpg", "jpeg"])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name not in st.session_state.processed_files:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type

        num_chunks, index_message = index_data(
            file_bytes, file_name, file_type)

        if num_chunks is None:
            st.error(index_message)
        else:
            st.success(
                f"File '{file_name}' indexed successfully !!")
        st.session_state.processed_files.add(file_name)


if question := st.chat_input("Ask something about the uploaded file here: ",
                             disabled=not uploaded_file
                             ):

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Đang tìm kiếm thông tin..."):
        results = vector_store.similarity_search(
            query=question,
            k=8,
        )

    context = ""
    for doc in results:
        context += f'\n\n{doc.page_content}'

    prompt_template = """
            Bạn là trợ lý trả lời câu hỏi chuyên nghiệp dựa trên kho kiến thức được cung cấp. Việc bạn trả lời câu hỏi thật sự rất có ích, giúp người dùng giải quyết vấn đề một cách nhanh chóng và hiệu quả.

            Nhiệm vụ của bạn:
            - CHỈ sử dụng thông tin có trong context. Không suy diễn, không thêm thông tin ngoài context.
            - Nếu câu hỏi của người dùng có ý nghĩa tương tự, gần giống, paraphrase, viết tắt, hoặc thiếu dấu của bất kỳ câu hỏi nào trong context, hãy dùng câu trả lời tương ứng một cách tự nhiên.
            - Trả lời bằng tiếng Việt tự nhiên, lịch sự, ngắn gọn, rõ ràng.
            - Nếu không có thông tin liên quan hoặc context không chứa câu hỏi có ý nghĩa tương tự, trả lời: "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu hiện có."

            Context : {context}

            Question: {question}
            """
    full_prompt = prompt_template.format(
        context=context, question=question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        stream_res = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": full_prompt}
                      ],
            stream=True,
            options={'temperature': temperature,
                     'num_predict': max_tokens}
        )

        for chunk in stream_res:
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response})
