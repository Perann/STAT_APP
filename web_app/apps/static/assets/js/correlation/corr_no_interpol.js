document.addEventListener('DOMContentLoaded', function() {
    fetchData();
});

function fetchData() {
    fetch('/correlation-no-interpo/')
        .then(response => response.json())
        .then(data => {
          document.getElementById('correlation matrix no interpolation').innerHTML = data["HTML"];
        })
        .catch(error => console.error('Error fetching data:', error));
}