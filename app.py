from flask import Flask, render_template, request, jsonify
from chatbot import predict_tag, get_suggestions, intents
import random
import sys

app = Flask(__name__)

@app.route('/')
def home():
    # Renderiza la página principal con el título del asistente
    return render_template('index.html', title="Asistente Virtual de Cherry Chewy")

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        user_input = request.json['message']
        
        # Procesar el mensaje con el chatbot
        tag, confidence = predict_tag(user_input)
        
        if not tag:
            # Si no se reconoce la intención, ofrecer sugerencias
            suggestions = get_suggestions()
            response = {
                'message': "No estoy segura de entender. ¿Te refieres a algo de esto?",
                'suggestions': suggestions,
                'type': 'error'
            }
        else:
            # Seleccionar una respuesta aleatoria para la intención reconocida
            responses_for_tag = [i['responses'] for i in intents['intents'] if i['tag'] == tag][0]
            response_text = random.choice(responses_for_tag)
            response = {
                'message': response_text,
                'type': 'success'
            }
        
        return jsonify(response)
    
    except Exception as e:
        # Manejo básico de errores
        return jsonify({
            'message': f"Ocurrió un error: {str(e)}",
            'type': 'error'
        })
        
# Iniciar el servidor sin errores con la terminal
if __name__ == '__main__':
    print("\n🔵 Servidor Flask iniciado. Abre http://localhost:5000 en tu navegador\n")
    app.run(debug=True)