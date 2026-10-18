// Voice narration using the browser's built-in speech synthesis (no network).
const synth = typeof window !== 'undefined' ? window.speechSynthesis : undefined;
const supported = !!synth && typeof SpeechSynthesisUtterance !== 'undefined';

let voice = null;
function pickVoice() {
  if (!supported) return;
  const voices = synth.getVoices().filter((v) => /^en[-_]/i.test(v.lang));
  const preferred = /Google|Natural|Neural|Samantha|Daniel|Aria|Jenny|Guy/i;
  voice = voices.find((v) => preferred.test(v.name)) ?? voices.find((v) => v.default) ?? voices[0] ?? null;
}
if (supported) {
  pickVoice();
  synth.addEventListener?.('voiceschanged', pickVoice);
}

// Plain text from lesson HTML, with symbols read in words.
export function speakable(html) {
  const div = document.createElement('div');
  div.innerHTML = html;
  div.querySelectorAll('h3').forEach((h) => { h.textContent = `${h.textContent}. `; });
  div.querySelectorAll('li').forEach((li) => { li.textContent = `${li.textContent}. `; });
  return div.textContent
    .replace(/[●→↑]/g, ' ')
    .replace(/≥/g, ' at least ')
    .replace(/≤/g, ' at most ')
    .replace(/\s+/g, ' ')
    .replace(/\.\s*\./g, '.')
    .trim();
}

// Chrome stops long utterances after ~15 s, so speak sentence by sentence.
const sentences = (text) => text.match(/[^.!?]+[.!?]*(\s|$)/g)?.map((s) => s.trim()).filter(Boolean) ?? [];

export const narrator = {
  supported,
  enabled: false,
  rate: 1,
  onChange: () => {},
  speak(text) {
    if (!supported || !this.enabled || !text) return;
    synth.cancel();
    const parts = sentences(text);
    parts.forEach((part, i) => {
      const u = new SpeechSynthesisUtterance(part);
      if (voice) u.voice = voice;
      u.rate = this.rate;
      if (i === parts.length - 1) u.onend = () => this.onChange(false);
      synth.speak(u);
    });
    this.onChange(true);
  },
  stop() {
    if (supported) synth.cancel();
    this.onChange(false);
  },
};
