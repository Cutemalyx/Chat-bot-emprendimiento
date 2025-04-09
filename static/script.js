document.addEventListener('DOMContentLoaded', function() {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    
    let currentSuggestions = []; // Almacena las sugerencias actuales
    
    // Mensaje inicial del bot
    addMessage('¡Hola! Soy el asistente de Cherry Chewy. ¿En qué puedo ayudarte?', 'bot');
    
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender + '-message');
        messageDiv.textContent = text;
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    
    function showSuggestions(suggestions) {
        currentSuggestions = suggestions; // Guarda las sugerencias actuales
        let suggestionsText = "No estoy segura de entender. ¿Te refieres a algo de esto?\n";
        suggestions.forEach((suggestion, index) => {
            suggestionsText += `\n${index + 1}. ${suggestion.charAt(0).toUpperCase() + suggestion.slice(1)}`;
        });
        suggestionsText += "\n\nResponde con el número o reformula tu pregunta ❓";
        addMessage(suggestionsText, 'bot');
    }
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Manejar comando "salir"
        if (message.toLowerCase() === 'salir') {
            addMessage('¡Gracias por visitar Cherry Chewy! 💕', 'bot');
            userInput.value = '';
            userInput.disabled = true;
            sendButton.disabled = true;
            return;
        }
        
        // Verificar si es una respuesta numérica a sugerencias
        if (currentSuggestions.length > 0 && /^\d+$/.test(message)) {
            const selectedNumber = parseInt(message);
            if (selectedNumber >= 1 && selectedNumber <= currentSuggestions.length) {
                // Enviar la sugerencia seleccionada en lugar del número
                const selectedSuggestion = currentSuggestions[selectedNumber - 1];
                addMessage(selectedNumber.toString(), 'user'); // Mostrar el número ingresado
                sendToBackend(selectedSuggestion);
                currentSuggestions = []; // Limpiar las sugerencias actuales
                userInput.value = '';
                return;
            }
        }
        
        addMessage(message, 'user');
        userInput.value = '';
        sendToBackend(message);
    }
    
    function sendToBackend(message) {
        // Enviar al backend
        fetch('/send_message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.type === 'error' && data.suggestions) {
                showSuggestions(data.suggestions);
            } else {
                addMessage(data.message, 'bot');
            }
        })
        .catch(error => {
            addMessage('Lo siento, hubo un error al procesar tu mensaje.', 'bot');
            console.error('Error:', error);
        });
    }
    
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});