import { app } from "../../scripts/app.js";

// MiniMaxModelAdvisor renders its report directly inside the node, in Markdown.
//
// The advisor's whole purpose is to be *read*: hardware line, one recommended file
// per task with its star rating, and the measured fit verdict.  A plain STRING
// output shows that as clipped raw text, so the node reuses the same Markdown
// text-preview widget the prompt report uses (ComfyUI only attaches it to its own
// PreviewAny / SaveText nodes).  The STRING output itself is unchanged and stays
// available for a downstream node or an app UI.

const NODE_TYPE = "MiniMaxModelAdvisor";
const PLAIN_OUTPUT_NAME = "report";
const PREVIEW_MODE_NAME = "preview_mode";

app.registerExtension({
    name: "minimax_music_production_toolkit.modelAdvisorPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return;

        const previewApi = window.comfyAPI?.textPreviewWidgets;
        if (!previewApi?.addTextPreviewWidgets || !previewApi?.updateTextPreviewWidgets) {
            console.warn(
                "[Music Production Toolkit] This ComfyUI frontend does not expose the Markdown text-preview API; the model advisor will show as plain text."
            );
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            try {
                const plain = this.widgets?.find((widget) => widget.name === PLAIN_OUTPUT_NAME);
                if (plain) {
                    plain.options.hidden = true;
                    plain.hidden = true;
                }
                previewApi.addTextPreviewWidgets(this);
                const toggle = this.widgets?.find((widget) => widget.name === PREVIEW_MODE_NAME);
                if (toggle) {
                    toggle.value = true; // Markdown mode by default
                    toggle.callback?.(true);
                }
                this.setDirtyCanvas?.(true, true);
            } catch (error) {
                console.warn(
                    "[Music Production Toolkit] Could not enable the Markdown preview for MiniMaxModelAdvisor:",
                    error
                );
            }
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            originalOnExecuted?.apply(this, arguments);
            try {
                previewApi.updateTextPreviewWidgets(this, output);
                const toggle = this.widgets?.find((widget) => widget.name === PREVIEW_MODE_NAME);
                if (toggle && toggle.value !== true) {
                    toggle.value = true;
                    toggle.callback?.(true);
                }
            } catch (error) {
                console.warn(
                    "[Music Production Toolkit] Could not update the Markdown preview for MiniMaxModelAdvisor:",
                    error
                );
            }
        };
    },
});
