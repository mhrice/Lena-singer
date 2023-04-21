from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from audio_learner import (
    load_model,
    parse_label,
    generate_audio,
    pre_generation_processing,
    post_generation_processing,
    LearnerStateModel,
    get_song_data,
    set_next_song,
    add_fades,
)
from flask_cors import CORS
import utils.utils as utils
import os
import numpy as np
from collections import deque
import time
import soundfile as sf
import shutil


# app = Flask(__name__)
app = Flask(__name__, static_folder="../build", static_url_path="/")
app.debug = True
app.secret_key = "random secret key!"
CORS(app)
cors = CORS(app, resource={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")
model_dir = "/Users/matthewrice/Developer/VISinger2/ckpt"
singer_model, hps = load_model(model_dir)
learner_model = LearnerStateModel(init_motivation=78, init_artistry=26)
prev_motivation = 1
prev_confidence = 1
streaming = False
should_stop = False


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def connect():
    global should_stop
    print("Client connected")
    shutil.rmtree("./web/public/audio", ignore_errors=True)
    try:
        os.mkdir("./web/public/audio")
    except FileExistsError:
        pass
    if streaming:
        should_stop = True
        time.sleep(1)


@socketio.on("disconnect")
def disconnect():
    print("Client disconnected")


@socketio.on("stop")
def stop():
    global should_stop
    should_stop = True


@socketio.on("stream_audio")
def stream_audio(data):
    global streaming
    global should_stop
    if not streaming:
        print("Streaming Audio...")
        prev_confidence = 1
        prev_motivation = 1
        learner_model = LearnerStateModel(
            init_motivation=data["motivation"], init_artistry=data["artistry"]
        )
        streaming = True
        is_paused = False
        while True:
            if should_stop:
                streaming = False
                should_stop = False
                break
            mistakes = learner_model.get_mistakes()
            confidence = learner_model.get_confidence()
            motivation = learner_model.get_motivation()
            artistry = learner_model.get_artistry()
            time_step = learner_model.get_time_step()
            mistake_memory = learner_model.get_mistake_memory()
            if (
                time_step == 30
                or (motivation < 1 and prev_motivation < 1)
                or (confidence > 99 and prev_confidence > 99)
            ):
                print("--------- FINISHED ---------")
                print(f"Motivation: {motivation}")
                print(f"Confidence: {confidence}")
                print(f"Artistry: {artistry}")
                print(f"Mistakes: {mistakes}")
                emit("finished")
                streaming = False
                break

            audio = learn_to_sing(mistakes, confidence)
            prev_motivation = motivation
            prev_confidence = confidence
            learner_model.step()

            print(f"--------- Time step: {time_step} ---------")
            print("Confidence", confidence)
            print("Motivation", motivation)
            print("Artistry", artistry)
            print("Mistakes", mistakes)

            chunk_size = 1024
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size] * 500
                emit("audio_chunk", chunk.tolist())
            emit(
                "audio_chunk_end",
                {
                    "confidence": float(confidence),
                    "motivation": float(motivation),
                    "mistakes": float(mistakes),
                    "artistry": float(artistry),
                    "mistake_memory": int(mistake_memory),
                    "time_step": int(time_step),
                },
            )
            time.sleep(0.5)
            # sf.write(
            #     f"./web/public/audio/{time_step}.wav",
            #     audio,
            #     hps.data.sample_rate,
            # )


@socketio.on("next_song")
def next_song():
    emit("song_title", set_next_song())


def learn_to_sing(mistakes, confidence):
    # Get next song data
    phones, notes, dur, slurs = get_song_data()
    # pre: mistakes from overall speed, phoneme speed change, random phoneme replacement
    phones, notes, dur = pre_generation_processing(phones, notes, dur, mistakes)
    phones, notes, dur, slurs = parse_label(hps, phones, notes, dur, slurs)
    # During: Detune mistakes
    audio = generate_audio(
        singer_model, phones, notes, dur, slurs, detune_prob=mistakes / 100
    )
    # Post: mistakes from filtering, distortion, compression. Confidence-controlled Reverb, Gain
    audio = post_generation_processing(
        audio, mistakes, confidence, hps.data.sample_rate
    )
    # Add fades between chunks
    learner_model.step()

    return audio


if __name__ == "__main__":
    socketio.run(app)
