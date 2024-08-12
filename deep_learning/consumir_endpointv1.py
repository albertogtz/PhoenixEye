<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flask API Client</title>
</head>
<body>
    <h1>Flask API Client</h1>
    <button id="predictButton">Get Prediction</button>
    <pre id="result"></pre>

    <script>
        document.getElementById('predictButton').addEventListener('click', async () => {
            const url = 'http://127.0.0.1:5000/predict';
            const data = {
                val_dataset: [
                    // Incluye aquí los datos necesarios en el formato esperado por el servidor
                    [[/* ... */], [/* ... */], [/* ... */], [/* ... */], [/* ... */]],
                    // Puedes añadir más secuencias si es necesario
                ],
                max_idx: 0
            };

            try {
                const response = await fetch(url, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                document.getElementById('result').textContent = JSON.stringify(result, null, 2);

                // Para manipular la matriz de pronóstico:
                const predictedFrame = result.predicted_frame;
                console.log('Predicted Frame:', predictedFrame);

                // Puedes hacer más cosas con la matriz de pronóstico aquí
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').textContent = `Error: ${error.message}`;
            }
        });
    </script>
</body>
</html>
