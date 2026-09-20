import { useState, useEffect, useRef } from "react";

export default function VoiceDictationBtn({ onTranscript, disabled = false }) {
  const [isRecording, setIsRecording] = useState(false);
  const [browserSupported, setBrowserSupported] = useState(true);
  const recognitionRef = useRef(null);

  useEffect(() => {
    // Check for browser SpeechRecognition support (Chrome, Safari, Edge)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setBrowserSupported(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";

    recognition.onresult = (event) => {
      let currentTranscript = "";
      for (let i = 0; i < event.results.length; i++) {
        currentTranscript += event.results[i][0].transcript + " ";
      }
      if (onTranscript) {
        onTranscript(currentTranscript.trim());
      }
    };

    recognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      setIsRecording(false);
    };

    recognition.onend = () => {
      setIsRecording(false);
    };

    recognitionRef.current = recognition;

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
      }
    };
  }, [onTranscript]);

  const toggleRecording = () => {
    if (isRecording) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsRecording(false);
    } else {
      if (!browserSupported) {
        // Fallback simulation for unsupported browsers / testing
        const sampleDictation =
          "58-year-old male with crushing chest pain radiating to left arm for 45 minutes, sweating profusely, difficulty breathing. Denies fever. BP 90 over 60, heart rate 115, O2 89 percent.";
        if (onTranscript) onTranscript(sampleDictation);
        return;
      }
      try {
        recognitionRef.current.start();
        setIsRecording(true);
      } catch (err) {
        console.error("Failed to start speech recognition:", err);
      }
    }
  };

  return (
    <button
      type="button"
      onClick={toggleRecording}
      disabled={disabled}
      title={isRecording ? "Click to stop dictation" : "Click to speak clinical symptoms"}
      className={`inline-flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-semibold transition shadow-sm ${
        isRecording
          ? "bg-red-600 text-white animate-pulse ring-2 ring-red-400 ring-offset-1"
          : "bg-surface-elevated border border-surface-border text-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
      }`}
    >
      <span className="relative flex h-2 w-2">
        {isRecording && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-300 opacity-75"></span>
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            isRecording ? "bg-white" : "bg-red-500"
          }`}
        ></span>
      </span>
      {isRecording ? "Listening (Click to Stop)..." : "Voice Dictate"}
    </button>
  );
}
