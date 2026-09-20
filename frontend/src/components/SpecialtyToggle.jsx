const SPECIALTIES = [
  { id: "general", label: "General ED" },
  { id: "peds", label: "Pediatrics" },
  { id: "trauma", label: "Trauma Hub" },
  { id: "cardiology", label: "Cardiology" },
  { id: "stroke", label: "Stroke Unit" },
];

export default function SpecialtyToggle({ selectedId, onSelect }) {
  return (
    <div className="space-y-2">
      <span className="text-xs font-bold text-surface-muted block uppercase tracking-wider">Filter Specialty Mix</span>
      <div className="flex flex-wrap gap-2">
        {SPECIALTIES.map((spec) => {
          const active = selectedId === spec.id;
          return (
            <button
              key={spec.id}
              onClick={() => onSelect(spec.id)}
              className={`px-4 py-2 rounded-xl text-xs font-semibold transition-all border ${
                active
                  ? "bg-accent-blue text-white shadow-sm border-accent-blue"
                  : "bg-accent-wash text-surface-muted hover:text-white border-surface-border"
              }`}
            >
              <span>{spec.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
