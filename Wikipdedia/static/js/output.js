function validateForm() {
    var language = document.getElementById("language").value;

    if (language === "Select") {
        alert("Please select a language.");
        return false;
    }
}

