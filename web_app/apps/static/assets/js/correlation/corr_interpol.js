document.addEventListener('DOMContentLoaded', function() {
    fetchData();
});

function fetchData() {
    fetch('/correlation-interpo/')
        .then(response => response.json())
        .then(data => {
          document.getElementById('correlation matrix interpolation').innerHTML = data["HTML"];
        })
        .catch(error => console.error('Error fetching data:', error));
}
