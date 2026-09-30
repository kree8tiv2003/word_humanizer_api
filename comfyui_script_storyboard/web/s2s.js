// Adds an "upload script" button to the S2S Load Script node.
// Files go to ComfyUI/input/scripts/ via ComfyUI's standard upload endpoint.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ACCEPT = ".fountain,.txt,.fdx,.pdf,.docx,.md,.spmd";

async function uploadScript(file) {
  const body = new FormData();
  body.append("image", file, file.name); // endpoint name is historical; it accepts any file
  body.append("subfolder", "scripts");
  body.append("type", "input");
  body.append("overwrite", "true");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

app.registerExtension({
  name: "Script2Storyboard.ScriptUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "S2S_LoadScript") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const fileWidget = this.widgets.find((w) => w.name === "script_file");
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ACCEPT;
      input.style.display = "none";
      document.body.append(input);
      input.onchange = async () => {
        if (!input.files.length) return;
        try {
          const name = await uploadScript(input.files[0]);
          const values = fileWidget.options.values.filter((v) => !v.startsWith("("));
          if (!values.includes(name)) values.push(name);
          fileWidget.options.values = values.sort();
          fileWidget.value = name;
          fileWidget.callback?.(name);
          app.graph.setDirtyCanvas(true);
        } catch (e) {
          alert(`Script upload failed: ${e.message}`);
        }
        input.value = "";
      };
      const btn = this.addWidget("button", "upload script", null, () => input.click(), { serialize: false });
      btn.serialize = false; // keep widgets_values identical to the Python widget list
      const onRemoved = this.onRemoved;
      this.onRemoved = function () {
        input.remove();
        return onRemoved?.apply(this, arguments);
      };
      return r;
    };
  },
});

// Show text returned by S2S nodes (stats, character bible, prompts, reports) inside the node.
app.registerExtension({
  name: "Script2Storyboard.ShowText",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData.name?.startsWith("S2S_")) return;
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const text = message?.text;
      if (!text) return;
      const value = Array.isArray(text) ? text.join("\n") : String(text);
      let w = this.widgets?.find((x) => x.name === "s2s_output");
      if (!w) {
        const el = document.createElement("textarea");
        el.readOnly = true;
        el.style.cssText = "width:100%;font:11px monospace;opacity:.85;resize:none;";
        w = this.addDOMWidget("s2s_output", "s2s_text", el, { serialize: false, getValue: () => el.value, setValue: () => {} });
        w.serialize = false;
        w.computeSize = (width) => [width, 160];
      }
      w.element.value = value;
      this.setSize([Math.max(this.size[0], 420), this.computeSize()[1]]);
      app.graph.setDirtyCanvas(true, true);
    };
  },
});
