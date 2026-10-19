import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("Cargando modelo . . .")

modelo = ChatterboxMultilingualTTS.from_pretrained(device = "cpu")

texto = "Hola, bienvenidos a IA generativa, vamos a crear audios, imagenes, texto y si el pc aguanta (XD) videos, apenas soy un modelo pequeño, NO ESPERAR GRANDES COSAS"

print("Generando AUDIO Ø")

audio = modelo.generate(texto, language_id = "es", exaggeration = 0.7, cfg_weight = 0.4, temperature = 0.8, repetition_penalty = 1.2, min_p = 0.05, top_p = 1.0)

ta.save ("audio.wav", audio, modelo.sr)
print("Audio guardado ®")
