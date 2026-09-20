import { useState } from "react";
import VoiceDictationBtn from "./VoiceDictationBtn.jsx";
import EntityChips from "./EntityChips.jsx";
import { parseSymptomsNLP } from "../../services/api.js";

const PRESET_CLINICAL_CASES = [
  {
    name: "Acute STEMI Case",
    text: "58-year-old male with severe crushing chest pain radiating to left arm for 45 minutes, sweating profusely, and difficulty breathing. Denies fever. BP 90/60, HR 115, O2 89%, pain 9/10.",
  },
  {
    name: "Acute Stroke (FAST+) Case",
    text: "67yo female presenting with sudden facial droop, right arm weakness, and slurred speech starting 40 minutes ago. Denies seizure or trauma. BP 175/100, HR 88, RR 18.",
  },
  {
    name: "Severe Sepsis Alert",
    text: "74-year-old male with high fever, rigors, lethargic and altered mental status. Denies chest pain. BP 82/50, HR 128, RR 26, temp 103.2 F, O2 91%.",
  },
  {
    name: "Stable / Routine Case",
    text: "28yo female with mild localized ankle sprain after minor slip. Denies head injury, no chest pain, no SOB. Vitals: BP 118/76, HR 72, RR 14, O2 99%, pain 3/10.",
  },
];

export default function SymptomNLPInput({ onApplyAutoFill, isApplying = false }) {
  const [text, setText] = useState("");
  const [nlpData, setNlpData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleExtract = async (textToParse = text) => {
    if (!textToParse || !textToParse.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await parseSymptomsNLP(textToParse);
      setNlpData(data);
      if (onApplyAutoFill && data.auto_fill_payload) {
        onApplyAutoFill(data.auto_fill_payload);
      }
    } catch (err) {
      console.error("NLP extraction error:", err);
      setError("Failed to parse symptoms. Please check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const handleTranscript = (dictatedText) => {
    setText((prev) => (prev ? `${prev} ${dictatedText}` : dictatedText));
    handleExtract(dictatedText);
  };

  const handleSelectPreset = (preset) => {
    setText(preset.text);
    handleExtract(preset.text);
  };

  return (
    <div className="panel p-6 border-l-4 border-l-indigo-600 dark:border-l-indigo-500 shadow-md">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3">
        <div>
          <h2 className="text-base font-bold text-surface-ink flex items-center gap-2">
            <span>🩺</span> Natural Language & Voice Intake (Clinical NLP)
          </h2>
          <p className="text-xs text-surface-muted">
            Dictate or type clinical notes. The NLP pipeline extracts vitals, findings, negations, and maps ICD-10 codes to auto-fill the triage form.
          </p>
        </div>
        <VoiceDictationBtn onTranscript={handleTranscript} disabled={loading} />
      </div>

      {/* Preset Quick-Test Cases */}
      <div className="flex flex-wrap items-center gap-1.5 pt-1 pb-3">
        <span className="text-[11px] font-medium text-slate-500">Quick Test Cases:</span>
        {PRESET_CLINICAL_CASES.map((c, i) => (
          <button
            key={i}
            type="button"
            onClick={() => handleSelectPreset(c)}
            className="rounded-full border border-slate-300 bg-white px-2.5 py-0.5 text-xs font-medium text-slate-700 hover:border-indigo-400 hover:bg-indigo-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 transition"
          >
            {c.name}
          </button>
        ))}
      </div>

      {/* Text Area */}
      <div className="relative">
        <textarea
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. 54yo male with crushing chest pain for 30 minutes, sweating and difficulty breathing. Denies fever. BP 90/60, HR 115, O2 89%."
          className="input w-full resize-none font-sans text-sm leading-relaxed"
        />
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-surface-muted">
          {text.trim() ? `${text.trim().split(/\s+/).length} words` : "Empty note"}
        </span>
        <button
          type="button"
          onClick={() => handleExtract()}
          disabled={loading || !text.trim()}
          className="btn-primary inline-flex items-center gap-2 text-xs font-semibold px-4 py-2"
        >
          {loading ? (
            <>
              <span className="animate-spin h-3.5 w-3.5 border-2 border-white border-t-transparent rounded-full" />
              Extracting NLP Entities…
            </>
          ) : (
            <>
              <span>✨</span> Parse & Auto-Fill Form
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="mt-3 rounded-lg bg-red-50 p-2.5 text-xs text-red-600 border border-red-200">
          {error}
        </div>
      )}

      {/* NLP Feedback Chips */}
      {nlpData && (
        <div className="mt-4">
          <EntityChips
            urgencyLevel={nlpData.urgency_level}
            urgencyKeywords={nlpData.urgency_keywords}
            entityChips={nlpData.entity_chips}
            icdMappings={nlpData.icd_mappings}
          />
        </div>
      )}
    </div>
  );
}
