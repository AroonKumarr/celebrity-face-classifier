// Make sure Dropzone doesn't auto-bind the <form> by itself
Dropzone.autoDiscover = false;

function init() {
    // Bind to your actual form id in app.html: id="dropzone"
    const dz = new Dropzone("#dropzone", {
        url: "/classify_image",     // DIRECTLY call classifier endpoint
        method: "post",
        paramName: "file",          // Flask expects request.files['file']
        maxFiles: 1,
        acceptedFiles: "image/*",
        autoProcessQueue: false,    // only upload when button clicked ✅ FIXED (added comma)
        addRemoveLinks: true,       // shows a remove button
        dictRemoveFile: "Remove"    // text for remove link
    });

    // ensure only one file
    dz.on("addedfile", function () {
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    // Handle server response (predictions)
    dz.on("success", function (file, response) {
        console.log("Classification result:", response);

        if (!response || response.length === 0) {
            $("#resultHolder").hide();
            $("#divClassTable").hide();
            $("#error").show();
            return;
        }

        // pick best match
        let match = null;
        let bestScore = -1;
        for (let i = 0; i < response.length; ++i) {
            const probs = response[i].class_probability || [];
            const maxScore = Math.max.apply(null, probs);
            if (maxScore > bestScore) {
                bestScore = maxScore;
                match = response[i];
            }
        }

        if (!match) {
            $("#resultHolder").hide();
            $("#divClassTable").hide();
            $("#error").show();
            return;
        }

        // Show result card on the left (use ui_key for HTML data-player)
        $("#error").hide();
        $("#resultHolder").show();
        $("#divClassTable").show();

        const uiKey = match.ui_key || match.class; // fallback
        $("#resultHolder").html($(`[data-player="${uiKey}"]`).html());

        // Fill probability table using UI-mapped dictionary when available
        const classDict = match.class_dictionary_ui || match.class_dictionary || {};
        for (const person in classDict) {
            const idx = classDict[person];
            const score = match.class_probability[idx] || 0;
            // IDs in HTML: score_babar, score_hania, ...
            const cellId = "#score_" + person;
            $(cellId).html(score.toFixed(2) + "%");
        }
    });

    // Manual trigger
    $("#submitBtn").on("click", function () {
        if (dz.files.length === 0) {
            alert("Please select an image to upload.");
            return;
        }
        // reset UI panels before new request
        $("#error").hide();
        $("#resultHolder").hide().empty();
        $("#divClassTable").show();
        $("#classTable td[id^='score_']").html("");
        dz.processQueue();
    });
}

$(document).ready(function () {
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});
