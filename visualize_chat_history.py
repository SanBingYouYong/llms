import os
import json
import streamlit as st

# Directory containing chat history
CHAT_HISTORY_DIR = "chat_history"

def load_chat_history():
    """Load all chat history JSON files from the directory."""
    chat_data = []
    for filename in os.listdir(CHAT_HISTORY_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(CHAT_HISTORY_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                chat_data.append(json.load(file))
    return chat_data

def load_chat_history(file_name):
    """Load a specific chat history JSON file."""
    filepath = os.path.join(CHAT_HISTORY_DIR, file_name)
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def main():
    """Streamlit app to visualize chat content."""
    st.title("Chat History Visualizer")

    # List available chat history files
    files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]
    if not files:
        st.warning("No chat history files found.")
        return

    # Sort files by modification time (latest first)
    files.sort(key=lambda f: os.path.getmtime(os.path.join(CHAT_HISTORY_DIR, f)), reverse=True)

    # Sidebar options
    st.sidebar.title("Options")

    # Default to the latest modified file
    default_file = files[0] if files else None

    # File selection in the sidebar
    selected_file = st.sidebar.selectbox("Select a chat history file", files, index=0 if default_file else -1)

    # Display mode toggle in the sidebar
    display_mode = st.sidebar.radio("Select display mode", ("Raw Text", "Markdown"), index=1)

    # Sidebar option to select pagination mode
    pagination_mode = st.sidebar.radio(
        "Pagination Mode",
        ("10 Messages per Page", "1 Conversation Round per Page"),
        index=1
    )

    if selected_file:
        chat_data = load_chat_history(selected_file)
        st.success(f"Loaded chat history from {selected_file}")

        if pagination_mode == "10 Messages per Page":
            # Pagination logic for 10 messages per page
            total_pages = (len(chat_data) + 9) // 10  # Calculate total pages
            page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

            # Display messages for the selected page
            start_index = (page - 1) * 10
            end_index = start_index + 10
            st.subheader(f"Messages (Page {page} of {total_pages})")
            for message in chat_data[start_index:end_index]:
                role = message.get("role", "unknown").capitalize()
                content = message.get("content", [])

                if role.lower() == "user" and isinstance(content, list):
                    text_content = "\n".join(entry.get("text", "") for entry in content if entry.get("type") == "text")
                else:
                    text_content = message.get("content", "No content")

                st.chat_message(role.lower()).text(text_content)

        else:  # "1 Conversation Round per Page"
            # Group messages into conversation rounds (user + assistant)
            conversation_rounds = []
            current_round = {}
            system_prompts = []

            for message in chat_data:
                role = message.get("role", "").lower()
                if role == "system":
                    system_prompts.append(message)  # Collect system prompts
                elif role == "user":
                    if current_round:  # If there's an incomplete round, add it to the list
                        conversation_rounds.append(current_round)
                    current_round = {"user": message, "system": system_prompts[-1] if system_prompts else None}
                elif role == "assistant":
                    if current_round:  # Add assistant response to the current round
                        current_round["assistant"] = message
                        conversation_rounds.append(current_round)
                        current_round = {}  # Reset for the next round

            # Handle any remaining incomplete round
            if current_round:
                conversation_rounds.append(current_round)

            # Pagination logic for 1 conversation round per page
            total_pages = len(conversation_rounds)
            page = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

            # Display the selected conversation round
            st.subheader(f"Conversation Round (Page {page} of {total_pages})")
            selected_round = conversation_rounds[page - 1]

            # Display system prompt in effect for the selected round
            system_message = selected_round.get("system")
            if system_message:
                with st.expander("System Prompt", expanded=False):
                    st.text(system_message.get("content", "No content"))

            # Display user message
            user_message = selected_round.get("user", {})
            content = user_message.get("content", [])
            if isinstance(content, list):
                text_content = "\n".join(entry.get("text", "") for entry in content if entry.get("type") == "text")
            else:
                text_content = user_message.get("content", "No content")
            st.chat_message("user").text(text_content)

            # Display assistant message
            assistant_message = selected_round.get("assistant", {})
            if assistant_message:
                role = assistant_message.get("role", "unknown").capitalize()
                content = assistant_message.get("content", "")
                provider = assistant_message.get("provider", "Not Recorded")
                model = assistant_message.get("model", "Not Recorded")
                timestamp = assistant_message.get("timestamp", "Not Recorded")
                image_paths = assistant_message.get("image_paths", [])

                if display_mode == "Raw Text":
                    with st.expander("Query Details", expanded=True):
                        st.text(f"📦 Provider: `{provider}`")
                        st.text(f"📘 Model: `{model}`")
                        st.text(f"⏰ Timestamp: `{timestamp}`")
                        st.text(f"🖼️ Images: {', '.join(image_paths) if image_paths else 'None'}")
                    st.chat_message(role.lower()).text(content)
                else:
                    with st.expander("Query Details", expanded=True):
                        st.markdown(f"**📦 Provider:** `{provider}`")
                        st.markdown(f"**📘 Model:** `{model}`")
                        st.markdown(f"**⏰ Timestamp:** `{timestamp}`")
                        st.markdown(f"**🖼️ Images:** {', '.join(image_paths) if image_paths else 'None'}")
                        if image_paths:
                            cols = st.columns(3)
                            for i, image_path in enumerate(image_paths):
                                with cols[i % 3]:
                                    if os.path.exists(image_path):
                                        st.image(image_path, use_container_width=True)
                                    else:
                                        st.text(image_path)
                        else:
                            st.text("No images available.")
                    st.chat_message(role.lower()).markdown(content)

if __name__ == "__main__":
    main()