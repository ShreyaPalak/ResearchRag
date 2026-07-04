import gradio as gr

from config import PORT
import rag_engine


def handle_upload(pdf_file):
    if pdf_file is None:
        return '⚠️ No file uploaded.'
    # With type='filepath', pdf_file is already a path string.
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    try:
        n_chunks = rag_engine.build_pipeline(pdf_path)
    except ValueError as exc:
        return f'⚠️ {exc}'
    return f'✅ Indexed {n_chunks} chunks — ready to ask questions.'


def handle_question(question):
    if not question.strip():
        return 'Please enter a question.'
    return rag_engine.ask(question)


with gr.Blocks(title='PDF Research Chatbot') as demo:
    gr.Markdown('## 📄 PDF Research Chatbot — RAG (Groq + FAISS)')
    pdf_input = gr.File(label='Upload PDF', file_types=['.pdf'], type='filepath')
    upload_status = gr.Textbox(label='Status', interactive=False)
    pdf_input.change(handle_upload, inputs=pdf_input, outputs=upload_status)

    question_input = gr.Textbox(label='Ask a question about the PDF')
    answer_output = gr.Textbox(label='Answer', interactive=False)
    question_input.submit(handle_question, inputs=question_input, outputs=answer_output)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=PORT, share=False)
