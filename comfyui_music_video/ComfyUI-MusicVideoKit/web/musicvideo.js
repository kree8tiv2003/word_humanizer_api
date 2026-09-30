import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BATCH = 50;
const ACCEPT = {
  images: ".png,.jpg,.jpeg,.webp,.bmp",
  audio: ".wav,.mp3,.flac,.ogg,.m4a,.aac,.opus",
  script: ".txt,.json,.csv,.md",
};
const EXT_KIND = {};
for (const [kind, exts] of Object.entries(ACCEPT)) exts.split(",").forEach((e) => (EXT_KIND[e] = kind));

function notify(summary, detail, severity = "info") {
  const toast = app.extensionManager?.toast;
  if (toast?.add) toast.add({ severity, summary, detail, life: 6000 });
  else alert(`${summary}\n${detail}`);
}

function projectOf(node) {
  return node.widgets?.find((w) => w.name === "project")?.value || "my_music_video";
}

function describe(s) {
  return `${s.images.length} images, ${s.audio.length} audio files, ${s.script.length} script files, ${s.rendered_segments} segments rendered`;
}

async function uploadFiles(node, kind, files) {
  files = [...files].sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  if (!files.length) return;
  const project = projectOf(node);
  let status = null;
  const batches = Math.ceil(files.length / BATCH);
  for (let b = 0; b < batches; b++) {
    const chunk = files.slice(b * BATCH, (b + 1) * BATCH);
    const body = new FormData();
    body.append("project", project);
    body.append("kind", kind);
    chunk.forEach((f) => body.append("files", f, f.name));
    notify(`Uploading ${kind}`, `Batch ${b + 1}/${batches} (${chunk.length} files) → ${project}`);
    const res = await api.fetchApi("/musicvideo/upload", { method: "POST", body });
    const json = await res.json();
    if (!res.ok) {
      notify("Upload failed", json.error || res.statusText, "error");
      return;
    }
    if (json.rejected?.length) notify("Skipped unsupported files", json.rejected.join(", "), "warn");
    status = json.status;
  }
  if (status) notify(`Project "${project}" updated`, describe(status), "success");
}

function pickAndUpload(node, kind) {
  const input = document.createElement("input");
  input.type = "file";
  input.multiple = kind !== "script";
  input.accept = ACCEPT[kind];
  input.onchange = () => uploadFiles(node, kind, input.files);
  input.click();
}

function addButton(node, label, cb) {
  const w = node.addWidget("button", label, null, cb);
  w.serialize = false;
  return w;
}

app.registerExtension({
  name: "MusicVideoKit.BatchUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "MVScriptBuilder") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      addButton(this, "📤 Upload storyboard images (50 per batch)", () => pickAndUpload(this, "images"));
      addButton(this, "🎵 Upload audio (full song or clips)", () => pickAndUpload(this, "audio"));
      addButton(this, "📝 Upload script (.txt/.json/.csv)", () => pickAndUpload(this, "script"));
      addButton(this, "🔎 Show project status", async () => {
        const res = await api.fetchApi(`/musicvideo/status?project=${encodeURIComponent(projectOf(this))}`);
        const s = await res.json();
        notify(`Project "${s.project}"`, describe(s));
      });
      addButton(this, "🗑 Clear project files…", async () => {
        const kind = prompt("Clear which files? Type: images, audio or script");
        if (!kind || !ACCEPT[kind.trim()]) return;
        if (!confirm(`Delete all ${kind} in project "${projectOf(this)}"?`)) return;
        const res = await api.fetchApi("/musicvideo/clear", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project: projectOf(this), kind: kind.trim() }),
        });
        const json = await res.json();
        notify("Cleared", json.status ? describe(json.status) : json.error, json.status ? "success" : "error");
      });
      return r;
    };

    // Drag & drop files straight onto the node; they're sorted into images/audio/script by extension.
    nodeType.prototype.onDragOver = function (e) {
      return !!e?.dataTransfer?.types?.includes?.("Files");
    };
    nodeType.prototype.onDragDrop = function (e) {
      const groups = {};
      for (const f of e.dataTransfer?.files || []) {
        const ext = "." + f.name.split(".").pop().toLowerCase();
        const kind = EXT_KIND[ext];
        if (kind) (groups[kind] ||= []).push(f);
      }
      if (!Object.keys(groups).length) return false;
      (async () => {
        for (const [kind, files] of Object.entries(groups)) await uploadFiles(this, kind, files);
      })();
      return true;
    };
  },
});
