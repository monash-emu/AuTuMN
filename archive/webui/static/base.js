// Show or Hide element with id
function toggle(id) {
    var element = document.getElementById(id);

    if (element.style.display === 'none' || element.style.display === '') {
        element.style.display = 'block';
    } else {
        element.style.display = 'none';
    }
}