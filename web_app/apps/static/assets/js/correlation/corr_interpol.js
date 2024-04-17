document.addEventListener('DOMContentLoaded', function() {
    fetchData_interpo();
});

function fetchData_interpo() {
    fetch('/correlation-interpo/')
        .then(response => response.json())
        .then(data => {
          document.getElementById('correlation matrix interpolation').innerHTML = data["HTML"];
        })
        .catch(error => console.error('Error fetching data:', error));
}
