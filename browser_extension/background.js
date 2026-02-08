// Chatterbox TTS - Browser Extension Background Script
// Creates right-click context menu for selected text

// Create context menu item on install
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "readWithChatterbox",
        title: "🔊 Read with Chatterbox",
        contexts: ["selection"]
    });
    console.log("Chatterbox context menu created.");
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "readWithChatterbox") {
        const selectedText = info.selectionText;

        if (!selectedText || selectedText.trim() === "") {
            console.warn("No text selected.");
            return;
        }

        console.log("Sending to Chatterbox:", selectedText.substring(0, 50) + "...");

        // Send to local HTTP server
        fetch("http://127.0.0.1:5679/read", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: selectedText })
        })
            .then(response => response.json())
            .then(data => {
                console.log("Chatterbox response:", data);
            })
            .catch(err => {
                console.error("Chatterbox not running or error:", err);
            });
    }
});
