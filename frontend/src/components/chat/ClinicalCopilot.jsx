import { useState, useRef, useEffect } from "react";
import { queryCopilotChat } from "../../services/api.js";

const QUICK_PROMPTS = [
  "Explain this triage recommendation",
  "Why is shock index high?",
  "What does MAP mean?",
  "Show STEMI protocol checklist",
  "What are the ESI triage levels?",
];

export default function ClinicalCopilot({ patientContext = null }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hello! I am your **JeevLok AI Copilot**.\n\nI can assist with interpreting Emergency Severity Index (ESI) criteria, calculating physiological indices (Shock Index, MAP), or reviewing emergency protocols.\n\nHow can I help you today?",
      sources: ["JeevLok AI Reference Standards"],
      disclaimer: "AI recommends. Clinician decides.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleSend = async (queryText = input) => {
    if (!queryText || !queryText.trim() || loading) return;

    const userMessage = { role: "user", content: queryText.trim() };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const historyPayload = messages.map((m) => ({ role: m.role, content: m.content }));
      const response = await queryCopilotChat({
        message: queryText.trim(),
        sessionId: "copilot_session_1",
        patientContext: patientContext,
        history: historyPayload,
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: response.reply,
          sources: response.sources || [],
          disclaimer: response.disclaimer || "AI recommends. Clinician decides.",
        },
      ]);
    } catch (err) {
      console.error("Copilot query failed:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I encountered an error connecting to the clinical knowledge base. Please verify backend availability.",
          sources: [],
          disclaimer: "AI recommends. Clinician decides.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 px-4 py-3 text-sm font-semibold text-white shadow-xl hover:from-indigo-700 hover:to-purple-700 transition transform hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:ring-offset-2"
      >
        <span className="text-lg">✨</span>
        <span>{isOpen ? "Close Copilot" : "Clinical Copilot"}</span>
      </button>

      {/* Slide-over Modal / Drawer */}
      {isOpen && (
        <div className="fixed bottom-20 right-6 z-50 flex h-[560px] w-96 max-w-[calc(100vw-2rem)] flex-col rounded-2xl border border-surface-border bg-white shadow-2xl dark:bg-slate-900 overflow-hidden animate-in fade-in slide-in-from-bottom-5 duration-200">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-surface-border bg-gradient-to-r from-indigo-50 via-white to-purple-50 px-4 py-3 dark:from-slate-800 dark:via-slate-900 dark:to-indigo-950/40">
            <div className="flex items-center gap-2">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white text-base shadow-sm">
                🩺
              </span>
              <div>
                <h3 className="text-sm font-bold text-surface-ink">Clinical Copilot</h3>
                <p className="text-[10px] text-surface-muted">RAG Clinical Knowledge Assistant</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-800"
            >
              ✕
            </button>
          </div>

          {/* Quick Prompts */}
          <div className="flex gap-1.5 overflow-x-auto border-b border-surface-border/60 bg-slate-50/70 px-3 py-2 dark:bg-slate-800/40 no-scrollbar">
            {QUICK_PROMPTS.map((prompt, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSend(prompt)}
                disabled={loading}
                className="whitespace-nowrap rounded-full border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:border-indigo-400 hover:bg-indigo-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300 transition"
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3.5 text-xs">
            {messages.map((m, idx) => (
              <div
                key={idx}
                className={`flex flex-col ${
                  m.role === "user" ? "items-end" : "items-start"
                }`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 leading-relaxed shadow-sm ${
                    m.role === "user"
                      ? "bg-indigo-600 text-white rounded-br-none"
                      : "bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-200 rounded-bl-none border border-surface-border/50"
                  }`}
                >
                  <div className="whitespace-pre-wrap">{m.content}</div>

                  {/* Sources tag */}
                  {m.sources?.length > 0 && (
                    <div className="mt-2 border-t border-slate-200/50 pt-1 text-[10px] text-slate-500 dark:text-slate-400">
                      <span className="font-semibold">Sources:</span>{" "}
                      {m.sources.join(", ")}
                    </div>
                  )}

                  {/* Clinical Disclaimer */}
                  {m.disclaimer && (
                    <div className="mt-1 text-[10px] italic text-slate-400">
                      "{m.disclaimer}"
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex items-center gap-2 text-slate-400 text-xs italic">
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
                Copilot is reviewing clinical knowledge…
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Footer */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="border-t border-surface-border p-2.5 bg-surface"
          >
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about vitals, ESI, or protocols…"
                className="input flex-1 text-xs py-2"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="btn-primary px-3 py-2 text-xs font-semibold"
              >
                Send
              </button>
            </div>
          </form>
        </div>
      )}
    </>
  );
}
