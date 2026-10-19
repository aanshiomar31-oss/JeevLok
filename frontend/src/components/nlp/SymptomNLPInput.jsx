import { useState } from "react";
import VoiceDictationBtn from "./VoiceDictationBtn.jsx";
import EntityChips from "./EntityChips.jsx";
import { parseSymptomsNLP, runLLMTriage } from "../../services/api.js";
import { IconBrain } from "../common/Icons.jsx";

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

const PRIORITY_BADGE_STYLES = {
  P1: "bg-red-600 text-white border-red-700 animate-pulse",
  P2: "bg-orange-500 text-white border-orange-600",
  P3: "bg-amber-400 text-slate-900 border-amber-500",
  P4: "bg-emerald-500 text-white border-emerald-600",
  P5: "bg-blue-500 text-white border-blue-600",
};

export default function SymptomNLPInput({ onApplyAutoFill, isApplying = false }) {
  const [text, setText] = useState("");
  const [nlpData, setNlpData] = useState(null);
  const [llmData, setLlmData] = useState(null);
  const [mode, setMode] = useState("llm"); // "llm" | "fast_nlp"
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPromptDebug, setShowPromptDebug] = useState(false);

  const handleRunTriage = async (textToParse = text) => {
    if (!textToParse || !textToParse.trim()) return;
    setLoading(true);
    setError(null);
    try {
      if (mode === "llm") {
        const triageResult = await runLLMTriage(textToParse);
        setLlmData(triageResult);
        setNlpData(null); // Clear fast NLP view
        if (onApplyAutoFill && triageResult.auto_fill_payload) {
          onApplyAutoFill(triageResult.auto_fill_payload);
        }
      } else {
        const data = await parseSymptomsNLP(textToParse);
        setNlpData(data);
        setLlmData(null); // Clear LLM view
        if (onApplyAutoFill && data.auto_fill_payload) {
          onApplyAutoFill(data.auto_fill_payload);
        }
      }
    } catch (err) {
      console.error("NLP/LLM extraction error:", err);
      setError("Failed to execute inference pipeline. Please verify backend connectivity.");
    } finally {
      setLoading(false);
    }
  };

  const handleTranscript = (dictatedText) => {
    setText((prev) => (prev ? `${prev} ${dictatedText}` : dictatedText));
    handleRunTriage(dictatedText);
  };

  const handleSelectPreset = (preset) => {
    setText(preset.text);
    handleRunTriage(preset.text);
  };

  return (
    <div className="panel p-6 border-l-4 border-l-indigo-600 dark:border-l-indigo-500 shadow-md">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3">
        <div>
          <h2 className="text-base font-bold text-surface-ink flex items-center gap-2">
            <span>🩺</span> LLM-Powered Patient Triage AI & NLP Intake
          </h2>
          <p className="text-xs text-surface-muted">
            Parses clinical context, extracts vitals & NegEx negations, prioritizes urgency (P1–P5), and orchestrates LangChain clinical prompts.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Pipeline Mode Selector */}
          <div className="inline-flex rounded-lg border border-slate-300 dark:border-slate-700 p-0.5 bg-slate-100 dark:bg-slate-800 text-xs">
            <button
              type="button"
              onClick={() => setMode("llm")}
              className={`px-2.5 py-1 rounded-md font-semibold transition ${
                mode === "llm"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-600 dark:text-slate-300 hover:text-indigo-600"
              }`}
            >
              ✨ LangChain LLM Pipeline
            </button>
            <button
              type="button"
              onClick={() => setMode("fast_nlp")}
              className={`px-2.5 py-1 rounded-md font-semibold transition ${
                mode === "fast_nlp"
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-600 dark:text-slate-300 hover:text-indigo-600"
              }`}
            >
              ⚡ Fast NLP Only
            </button>
          </div>
          <VoiceDictationBtn onTranscript={handleTranscript} disabled={loading} />
        </div>
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
          placeholder="e.g. 54yo male with crushing chest pain for 30 minutes, sweating profusely and difficulty breathing. Denies fever. BP 90/60, HR 115, O2 89%."
          className="input w-full resize-none font-sans text-sm leading-relaxed"
        />
      </div>

      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-surface-muted">
          {text.trim() ? `${text.trim().split(/\s+/).length} words` : "Empty note"}
        </span>
        <button
          type="button"
          onClick={() => handleRunTriage()}
          disabled={loading || !text.trim()}
          className="btn-primary inline-flex items-center gap-2 text-xs font-semibold px-4 py-2"
        >
          {loading ? (
            <>
              <span className="animate-spin h-3.5 w-3.5 border-2 border-white border-t-transparent rounded-full" />
              {mode === "llm" ? "Orchestrating LangChain Pipeline…" : "Extracting NLP Entities…"}
            </>
          ) : (
            <>
              <IconBrain className="w-4 h-4" />
              {mode === "llm" ? "Run LangChain LLM Triage & Auto-Fill" : "Parse Symptoms & Auto-Fill"}
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="mt-3 rounded-lg bg-red-50 p-2.5 text-xs text-red-600 border border-red-200">
          {error}
        </div>
      )}

      {/* LLM Triage Decision Support Card */}
      {llmData && (
        <div className="mt-4 rounded-xl border border-indigo-200 bg-indigo-50/40 dark:border-indigo-900/60 dark:bg-indigo-950/20 p-4 transition">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-indigo-100 dark:border-indigo-900/40 pb-3">
            <div className="flex items-center gap-2.5">
              <span
                className={`px-3 py-1 rounded-full text-xs font-extrabold uppercase tracking-wide border shadow-sm ${
                  PRIORITY_BADGE_STYLES[llmData.priority] || "bg-indigo-600 text-white"
                }`}
              >
                {llmData.priority} — {llmData.urgency_level}
              </span>
              <span className="text-xs font-bold text-slate-800 dark:text-slate-100">
                {llmData.summary_headline}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-500">Risk Score:</span>
              <span className="font-mono font-bold text-indigo-700 dark:text-indigo-400">
                {llmData.risk_score}/100
              </span>
              <span className="text-slate-400">|</span>
              <span className="text-slate-500">Confidence:</span>
              <span className="font-semibold text-slate-700 dark:text-slate-300">
                {Math.round(llmData.confidence * 100)}%
              </span>
            </div>
          </div>

          {/* Clinical Rationale & Hemodynamics */}
          <div className="mt-3 text-xs leading-relaxed text-slate-700 dark:text-slate-300">
            <p className="font-medium">{llmData.clinical_rationale}</p>
          </div>

          {/* Physiological Indices & Red Flags */}
          {llmData.physiological_drivers && llmData.physiological_drivers.length > 0 && (
            <div className="mt-3 rounded-lg bg-white/80 dark:bg-slate-900/80 p-2.5 border border-slate-200 dark:border-slate-800">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                Physiological Drivers & Hemodynamics:
              </span>
              <ul className="mt-1 space-y-1 text-xs">
                {llmData.physiological_drivers.map((drv, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-slate-700 dark:text-slate-300">
                    <span className="text-amber-500 font-bold">•</span>
                    <span>{drv}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Actionable Clinical Protocols */}
          {llmData.recommended_protocols && llmData.recommended_protocols.length > 0 && (
            <div className="mt-3">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-600 dark:text-slate-400">
                Actionable Emergency Protocols Triggered:
              </span>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {llmData.recommended_protocols.map((proto, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-md bg-indigo-100 text-indigo-800 dark:bg-indigo-900/50 dark:text-indigo-200 px-2.5 py-1 text-xs font-semibold border border-indigo-300 dark:border-indigo-700"
                  >
                    <span>⚡</span> {proto}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Form Auto-Fill Confirmation & Prompt Inspector Toggle */}
          <div className="mt-4 pt-3 border-t border-indigo-100 dark:border-indigo-900/40 flex flex-wrap items-center justify-between gap-2">
            <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold flex items-center gap-1">
              ✓ Triage form automatically populated with extracted vitals & findings
            </span>
            <button
              type="button"
              onClick={() => setShowPromptDebug((prev) => !prev)}
              className="text-xs text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-300 font-medium underline flex items-center gap-1"
            >
              {showPromptDebug ? "Hide Prompt Orchestration" : "Inspect LangChain Prompt Orchestration"}
            </button>
          </div>

          {/* Prompt Orchestration Diagnostics */}
          {showPromptDebug && llmData.prompt_orchestration && (
            <div className="mt-3 rounded-lg bg-slate-900 text-slate-100 p-3 text-xs font-mono border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-slate-400 border-b border-slate-800 pb-1">
                <span>Framework: {llmData.prompt_orchestration.framework}</span>
                <span>Model: {llmData.prompt_orchestration.model_name}</span>
                <span>Latency: {llmData.prompt_orchestration.latency_ms} ms</span>
              </div>
              <div>
                <span className="text-amber-400 font-bold block mb-1">Orchestrated Prompt Input:</span>
                <pre className="whitespace-pre-wrap text-[11px] bg-slate-950 p-2 rounded border border-slate-800 text-slate-300 overflow-x-auto max-h-48">
                  {llmData.prompt_orchestration.orchestrated_human_prompt}
                </pre>
              </div>
              <div className="text-[11px] text-slate-400">
                Fallback Status:{" "}
                <span className={llmData.prompt_orchestration.fallback_used ? "text-amber-400 font-bold" : "text-emerald-400 font-bold"}>
                  {llmData.prompt_orchestration.fallback_used ? "Deterministic Fallback Active (Air-Gapped / Zero-API Mode)" : "Live Cloud LLM"}
                </span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Fast NLP Feedback Chips */}
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

