import Head from 'next/head'
import { Inter } from 'next/font/google'
import styles from '@/styles/Home.module.css'
import { AiOutlinePlayCircle, AiOutlinePauseCircle } from 'react-icons/ai'
import { useEffect, useState } from 'react'
import { Knob } from "react-rotary-knob";
import skin from "../skins/s12";
import { io } from 'socket.io-client';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

let audioContext: AudioContext;
let audioBuffers: AudioBuffer[] = [];
let audioBuffer: AudioBuffer;

let writePointer: number;
let socket: SocketIOClient.Socket;
let firstPlay = true;
const inter = Inter({ subsets: ['latin'] })
const initialActive: number[] = []

// function int16ToFloat32(inputArray: Int16Array, startIndex: number, length: number) {
//   var output = new Float32Array(inputArray.length - startIndex);
//   for (var i = startIndex; i < length; i++) {
//     var int = inputArray[i];
//     // If the high bit is on, then it is a negative number, and actually counts backwards.
//     var float = (int >= 0x8000) ? -(0x10000 - int) / 0x8000 : int / 0x7FFF;
//     output[i] = float;
//   }
//   return output;
// }

function int16ToFloat32(inputArray: Int16Array) {
  var output = new Float32Array(inputArray.length);
  for (var i = 0; i < inputArray.length; i++) {
    var int = inputArray[i];
    // If the high bit is on, then it is a negative number, and actually counts backwards.
    var float = (int >= 0x8000) ? -(0x10000 - int) / 0x8000 : int / 0x7FFF;
    output[i] = float;
  }
  return output;
}


let labels = Array.from(Array(30).keys());
let confidences: number[] = [];
let motivations: number[] = [];
let abilities: number[] = [];
let mistakes: number[] = [];

const graphOptions = {
  scales: {
    y:
    {
      min: 0,
      max: 100,
    },
    x:
    {

    },
  },
  animation: false,
};

export default function Home() {
  const [play, setPlay] = useState(0)
  const [ready, setReady] = useState(0)
  const [motivation, setMotivation] = useState(50)
  const [ability, setAbility] = useState(50)
  const [confidence, setConfidence] = useState(30)
  const [timestep, setTimestep] = useState(-1)
  const [active, setActive] = useState(initialActive)
  const [mouthOpen, setMouthOpen] = useState(0)
  const [lenaText, setLenaText] = useState("")
  const [currentSong, setCurrentSong] = useState("")



  useEffect(() => {
    audioContext = new (window.AudioContext ||
      window.webkitAudioContext)();
    // connect without CORS
    socket = io('http://127.0.0.1:5001/')
    socket.on('connect', () => {
      console.log('Successfully connected!');
      // Clear audioBuffer
      audioBuffers = []
      confidences = []
      motivations = []
      abilities = []
      mistakes = []
      writePointer = 0
      setReady(0);
      setActive([]);
      audioBuffer = audioContext.createBuffer(1, 44100 * 10, 44100);
      firstPlay = true;
      setLenaText("Hello, I am Lena. I am trying to learn how to sing. \nClick 'Begin' to get me started.")
      setCurrentSong("雨打湿了天空灰得更彻底")
    });

    socket.on('song_title', (data: string) => {
      setCurrentSong(data)
    })

    socket.on('audio_chunk', function (chunk: Int16Array) {
      console.log("AUDIO CHUNK RECEIVED")
      audioBuffer.copyToChannel(int16ToFloat32(chunk), 0, writePointer)
      writePointer += chunk.length;

    });

    socket.on("audio_chunk_end", function (data: any) {
      console.log("AUDIO CHUNK END RECEIVED")
      setConfidence(data.confidence)
      setAbility(data.artistry)
      setMotivation(data.motivation)
      setTimestep(data.timestep)

      confidences.push(data.confidence)
      motivations.push(data.motivation)
      abilities.push(data.artistry)
      mistakes.push(data.mistakes)
      if (data.confidence > 65)
        setLenaText("I am getting better at this!")
      else if (data.confidence < 35)
        setLenaText("I am getting worse at this!")
      let smallerBuffer = audioContext.createBuffer(1, writePointer, 44100);
      smallerBuffer.copyToChannel(audioBuffer.getChannelData(0), 0, 0)
      audioBuffers.push(smallerBuffer)
      writePointer = 0
      audioBuffer = audioContext.createBuffer(1, 44100 * 10, 44100);

      setActive([...active, 0])

    })

    socket.on("finished", () => {
      console.log("FINISHED")
      if (motivation < 2) {
        setMotivation(0)
        setLenaText("This is too tough. I will give up now.")
      }
      else if (confidence > 98) {
        setConfidence(100)
        setLenaText("I am a great singer! I am done learning.")

      }
      else {
        setLenaText("I am done learning for now.")

      }

      setReady(1)
      setPlay(0)
      firstPlay = true;



    });


  }, [])



  const playAudio = (index: number, playPause: number) => {
    if (playPause) {
      setMouthOpen(1)
      let source = audioContext.createBufferSource();
      source.connect(audioContext.destination);
      source.buffer = audioBuffers[index];
      // set active true
      setActive([...active.slice(0, index), 1, ...active.slice(index + 1)])
      source.start();
      source.onended = () => {
        setActive([...active.slice(0, index), 0, ...active.slice(index + 1)])
        // If nothing active, close mouth
        if (active.filter((x) => x === 1).length === 0)
          setMouthOpen(0)
      }

    }
  }


  const handlePlayToggle = () => {
    setPlay(play === 0 ? 1 : 0)
    console.log("Play toggled")
    if (play === 0) {
      if (ready) {
        setReady(0);
        confidences = []
        motivations = []
        abilities = []
        mistakes = [];
        writePointer = 0
        setReady(0);
        setActive([]);
        audioBuffer = audioContext.createBuffer(1, 44100 * 10, 44100);
        audioBuffers = [];
      }
      setLenaText("Let's start!")
      socket.emit('stream_audio', { motivation: motivation, artistry: ability });

    }
    else {
      socket.emit('pause');
      firstPlay = true;
    }

  }

  const handleNextSong = () => {
    setReady(0);
    confidences = []
    motivations = []
    abilities = []
    mistakes = [];
    setLenaText("New song? Okay!")
    setPlay(0)
    firstPlay = true;
    socket.emit('next_song');
    setActive([]);
    audioBuffer = audioContext.createBuffer(1, 44100 * 10, 44100);
    audioBuffers = [];
  }



  let data = {
    labels: labels,
    datasets: [
      {
        label: "Confidence",
        data: confidences,
        fill: true,
        borderColor: 'rgb(42, 116, 212)',
        backgroundColor: 'rgba(42, 116, 212, 0.5)',
      },
      {
        label: "Motivation",
        data: motivations,
        fill: true,
        borderColor: 'rgb(61, 166, 63)',
        backgroundColor: 'rgba(61, 166, 63, 0.5)',
      },
      {
        label: "Ability",
        data: abilities,
        fill: true,
        borderColor: 'rgb(173, 75, 219)',
        backgroundColor: 'rgba(173, 75, 219, 0.5)',
      },
      {
        label: "Mistakes",
        data: mistakes,
        fill: true,
        borderColor: 'rgba(245, 12, 20)',
        backgroundColor: 'rgba(245, 12, 20, 0.5)',
      }
    ],

  }
  let confidenceBarHeight = confidence + "%"
  let offset = 1.5;
  return (
    <>
      <Head>
        <title>Singer Learner</title>
        <meta name="description" content="Generated by create next app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      {(play == 1 || ready === 1) &&
        <>
          <Line data={data} options={graphOptions} className={styles.graph} />
          {audioBuffers.map((buffer, index) => {
            return (
              <div className={styles.audioPlayer} key={index} style={{ left: `${index / 30 * 100 + offset}%` }}>
                <AiOutlinePlayCircle size={25} onClick={() => playAudio(index, 1)} />
              </div>
            )
          })}
        </>
      }
      <main className={styles.main}>
        <div className={styles.description}>
          <div className={styles.play}>

            {play === 0 ? (
              <div className={styles.playButton} onClick={handlePlayToggle}>Begin</div>
            ) : (
              <div className={styles.pauseButton} onClick={handlePlayToggle}>Stop</div>
            )}
          </div>
          <div className={styles.buttons}>
            <div className={styles.knobContainer}>
              Motivation
              <Knob
                style={{ display: "inline-block" }}
                min={0}
                max={100}
                skin={skin}
                onChange={setMotivation}
                value={motivation}
                preciseMode={false}
                unlockDistance={0}
                className={styles.knob}
              />
            </div>
            <div className={styles.knobContainer}>
              Ability
              <Knob
                min={0}
                max={100}
                onChange={setAbility}
                value={ability}
                skin={skin}
                preciseMode={false}
                unlockDistance={0}
                className={styles.knob}
              />
            </div>
            <div className={styles.songContainer}>
              <div className={styles.songChoice} onClick={handleNextSong}>
                Next Song
              </div>
              <div className={styles.currentSong}>
                <div>Current Song:</div>
                <div>{currentSong}</div>
              </div>
            </div>
          </div>
        </div>

        <div className={styles.center}>
          <div className={styles.mouthContainer}>
            <div className={styles.eyes}>
              <div className={styles.eye}></div>
              <div className={styles.eyeTwo}></div>
            </div>
            {mouthOpen === 0 ? (
              <div className={styles.mouthClosed} />
            ) : (
              <div className={styles.mouthOpen}>
                <div className={styles.lezu}></div>
              </div>
            )
            }
          </div>
          <div className={styles.confidence}>
            Confidence
            <div className={styles.confidenceBar}>
              <div className={styles.confidenceBarFill} style={{ "height": confidenceBarHeight }}></div>
            </div>
          </div>
        </div>
        <div className={styles.textBox}>
          {lenaText}
        </div>

      </main>
    </>
  )
}
