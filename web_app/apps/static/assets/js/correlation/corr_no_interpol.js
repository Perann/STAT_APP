document.addEventListener('DOMContentLoaded', function() {
    fetchData_no_interpo();
});

function fetchData_no_interpo() {
    fetch('/correlation-no-interpo/')
        .then(response => response.json())
        .then(data => {
          document.getElementById('correlation matrix no interpolation').innerHTML = data["HTML"];
        })
        .catch(error => console.error('Error fetching data:', error));
}