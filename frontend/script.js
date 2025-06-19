const API_BASE_URL = 'http://localhost:5000';

function switchTab(tabName) {
   document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
   document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

   event.target.classList.add('active');
   document.getElementById(tabName + 'Tab').classList.add('active');

   resetSingleAnalysis();
}

async function analyzeSingleComment() {
   const comment = document.getElementById('commentInput').value.trim();

   if (!comment) {
      showError('singleError', 'Por favor, digite um comentário para análise.');
      return;
   }

   hideError('singleError');
   showLoading('singleLoading');
   hideResults('singleResults');

   try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
         method: 'POST',
         headers: {
            'Content-Type': 'application/json',
         },
         body: JSON.stringify({ comment: comment })
      });

      if (!response.ok) {
         throw new Error(`Erro na API: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();

      hideLoading('singleLoading');
      showSingleResult(result);

   } catch (error) {
      hideLoading('singleLoading');
      console.error('Erro:', error);

      if (error.name === 'TypeError' && error.message.includes('fetch')) {
         showError('singleError', 'Não foi possível conectar com o servidor. Verifique se a API está rodando.');
      } else {
         showError('singleError', `Erro ao processar o comentário: ${error.message}`);
      }
   }
}

function showSingleResult(result) {
   const isHate = result.is_hate_speech;
   const icon = document.getElementById('singleResultIcon');
   const text = document.getElementById('singleResultText');
   const confidence = document.getElementById('singleConfidence');
   const comment = document.getElementById('singleComment');

   if (isHate) {
      icon.innerHTML = '<i class="fas fa-exclamation-triangle hate-speech"></i>';
      icon.className = 'result-icon hate-speech';
      text.textContent = 'Discurso de Ódio Detectado';
      text.className = 'result-text hate-speech';
   } else {
      icon.innerHTML = '<i class="fas fa-check-circle not-hate-speech"></i>';
      icon.className = 'result-icon not-hate-speech';
      text.textContent = 'Comentário Seguro';
      text.className = 'result-text not-hate-speech';
   }

   confidence.innerHTML = `<i class="fas fa-chart-line"></i> Confiança: ${result.confidence}%`;
   comment.textContent = `"${result.comment}"`;

   showResults('singleResults');
}

function showLoading(loadingId) {
   document.getElementById(loadingId).style.display = 'block';
}

function hideLoading(loadingId) {
   document.getElementById(loadingId).style.display = 'none';
}

function showResults(resultsId) {
   document.getElementById(resultsId).style.display = 'block';
}

function hideResults(resultsId) {
   document.getElementById(resultsId).style.display = 'none';
}

function showError(errorId, message) {
   const errorElement = document.getElementById(errorId);
   errorElement.textContent = message;
   errorElement.style.display = 'block';
}

function hideError(errorId) {
   document.getElementById(errorId).style.display = 'none';
}

function resetSingleAnalysis() {
   document.getElementById('commentInput').value = '';
   hideResults('singleResults');
   hideError('singleError');
   hideLoading('singleLoading');
}

document.addEventListener('DOMContentLoaded', () => {
   const container = document.querySelector('.container');
   container.style.opacity = '0';
   container.style.transform = 'translateY(20px)';

   setTimeout(() => {
      container.style.transition = 'all 0.6s ease';
      container.style.opacity = '1';
      container.style.transform = 'translateY(0)';
   }, 100);
});