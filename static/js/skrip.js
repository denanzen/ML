document.getElementById("predict-form").addEventListener("submit", function (event) {
    event.preventDefault();
  
    // Ambil data dari form
    const data = {
      kesehatan_pohon: document.getElementById("kesehatan_pohon").value,
      kondisi_tanah: document.getElementById("kondisi_tanah").value,
      kualitas_air: document.getElementById("kualitas_air").value,
      tingkat_gangguan: document.getElementById("tingkat_gangguan").value,
    };
  
    // Kirim data ke server Flask menggunakan fetch
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    })
      .then((response) => response.json())
      .then((result) => {
        const resultDiv = document.getElementById("result");
        if (result.prediction) {
          resultDiv.innerHTML = `<h3>Prediksi Kesehatan Hutan: ${result.prediction}</h3>`;
        } else if (result.error) {
          resultDiv.innerHTML = `<p>Error: ${result.error}</p>`;
        }
      })
      .catch((error) => {
        document.getElementById("result").innerHTML = `<p>Error: ${error.message}</p>`;
      });
  });
  