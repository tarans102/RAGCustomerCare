
document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // 1. DOM ELEMENTS
    // ==========================================

    const chatForm = document.getElementById("chatForm");
    const messageInput = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");

    const messages = document.getElementById("messages");
    const chatArea = document.getElementById("chatArea");

    const welcomeScreen = document.getElementById("welcomeScreen");
    const newChatBtn = document.getElementById("newChatBtn");

    const API_URL = "/ask";

    let isProcessing = false;


    // ==========================================
    // 2. AUTO RESIZE TEXTAREA
    // ==========================================

    messageInput.addEventListener("input", () => {

        messageInput.style.height = "auto";

        messageInput.style.height =
            Math.min(messageInput.scrollHeight, 130) + "px";

    });


    // ==========================================
    // 3. SCROLL TO BOTTOM
    // ==========================================

    function scrollToBottom() {

        chatArea.scrollTo({
            top: chatArea.scrollHeight,
            behavior: "smooth"
        });

    }


    // ==========================================
    // 4. HIDE WELCOME SCREEN
    // ==========================================

    function hideWelcomeScreen() {

        if (welcomeScreen) {
            welcomeScreen.style.display = "none";
        }

    }


    // ==========================================
    // 5. CREATE MESSAGE ELEMENT
    // ==========================================

    function addMessage(text, sender, metadata = "") {

        const messageElement = document.createElement("div");

        messageElement.className = `message ${sender}`;

        // Avatar
        if (sender === "assistant") {

            const avatar = document.createElement("div");

            avatar.className = "message-avatar";
            avatar.textContent = "AI";

            messageElement.appendChild(avatar);

        }

        // Message content
        const content = document.createElement("div");

        content.className = "message-content";

        // Message bubble
        const bubble = document.createElement("div");

        bubble.className = "message-bubble";

        // Use textContent to prevent HTML injection.
        bubble.textContent = text;

        content.appendChild(bubble);

        // Optional metadata
        if (metadata) {

            const meta = document.createElement("div");

            meta.className = "message-meta";
            meta.textContent = metadata;

            content.appendChild(meta);

        }

        messageElement.appendChild(content);

        messages.appendChild(messageElement);

        scrollToBottom();

        return messageElement;

    }


    // ==========================================
    // 6. TYPING INDICATOR
    // ==========================================

    function showTypingIndicator() {

        const typingElement = document.createElement("div");

        typingElement.className = "message assistant";
        typingElement.id = "typingIndicator";

        const avatar = document.createElement("div");

        avatar.className = "message-avatar";
        avatar.textContent = "AI";

        const content = document.createElement("div");

        content.className = "message-content";

        const typing = document.createElement("div");

        typing.className = "typing";

        for (let i = 0; i < 3; i++) {

            const dot = document.createElement("span");

            typing.appendChild(dot);

        }

        content.appendChild(typing);

        typingElement.appendChild(avatar);
        typingElement.appendChild(content);

        messages.appendChild(typingElement);

        scrollToBottom();

    }


    function removeTypingIndicator() {

        const indicator =
            document.getElementById("typingIndicator");

        if (indicator) {
            indicator.remove();
        }

    }


    // ==========================================
    // 7. SEND QUESTION TO BACKEND
    // ==========================================

    async function sendMessage(question) {

        question = question.trim();

        if (!question || isProcessing) {
            return;
        }

        // Hide welcome screen
        hideWelcomeScreen();

        // Display customer message
        addMessage(question, "user");

        // Clear input
        messageInput.value = "";

        messageInput.style.height = "auto";

        // Set processing state
        isProcessing = true;

        sendBtn.disabled = true;

        showTypingIndicator();

        try {

            const response = await fetch(API_URL, {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question: question
                })

            });


            // Handle unsuccessful HTTP responses
            if (!response.ok) {

                let errorMessage =
                    "Something went wrong. Please try again.";

                try {

                    const errorData = await response.json();

                    if (errorData.detail) {
                        errorMessage = errorData.detail;
                    }

                } catch (error) {

                    // Keep default error message.

                }

                throw new Error(errorMessage);

            }


            // Parse backend response
            const data = await response.json();

            removeTypingIndicator();


            // Display AI answer
            const metadata = [
                data.intent
                    ? `Category: ${data.intent.replace(/_/g, " ")}`
                    : "",

                typeof data.response_time_ms === "number"
                    ? `Response time: ${data.response_time_ms} ms`
                    : "",

                typeof data.retries === "number"
                    ? `Retries: ${data.retries}`
                    : ""

            ].filter(Boolean).join(" • ");


            addMessage(
                data.answer || "I couldn't generate a response.",
                "assistant",
                metadata
            );


        } catch (error) {

            removeTypingIndicator();

            console.error("API Error:", error);

            addMessage(
                "Sorry, I couldn't connect to the support service. " +
                "Please check your connection and try again.",
                "assistant"
            );

        } finally {

            isProcessing = false;

            sendBtn.disabled = false;

            messageInput.focus();

        }

    }


    // ==========================================
    // 8. FORM SUBMISSION
    // ==========================================

    chatForm.addEventListener("submit", (event) => {

        event.preventDefault();

        const question = messageInput.value;

        sendMessage(question);

    });


    // ==========================================
    // 9. ENTER KEY SUPPORT
    // ==========================================

    messageInput.addEventListener("keydown", (event) => {

        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.isComposing
        ) {

            event.preventDefault();

            chatForm.requestSubmit();

        }

    });


    // ==========================================
    // 10. QUICK QUESTION BUTTONS
    // ==========================================

    const suggestionButtons =
        document.querySelectorAll("[data-question]");

    suggestionButtons.forEach((button) => {

        button.addEventListener("click", () => {

            const question =
                button.getAttribute("data-question");

            sendMessage(question);

        });

    });


    // ==========================================
    // 11. NEW CONVERSATION
    // ==========================================

    newChatBtn.addEventListener("click", () => {

        if (isProcessing) {
            return;
        }

        // Clear messages
        messages.innerHTML = "";

        // Clear input
        messageInput.value = "";

        messageInput.style.height = "auto";

        // Restore welcome screen
        welcomeScreen.style.display = "";

        // Reset scroll position
        chatArea.scrollTo({
            top: 0,
            behavior: "smooth"
        });

        messageInput.focus();

    });


    // ==========================================
    // 12. INITIALIZATION
    // ==========================================

    console.log("SupportDesk frontend initialized.");

});
