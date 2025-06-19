const API_BASE_URL = 'http://localhost:5000';

function switchTab(tabName) {
   document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
   document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

   event.target.classList.add('active');
   document.getElementById(tabName + 'Tab').classList.add('active');

   resetSingleAnalysis();
   resetBatchAnalysis();
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
      const response = await fetch(`${API_BASE_URL}/predict`, {
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

async function analyzeBatchComments() {
   const twitterUrl = document.getElementById('twitterInput').value.trim();

   if (!twitterUrl) {
      showError('batchError', 'Por favor, insira o link do tweet.');
      return;
   }

   if (!twitterUrl.includes('twitter.com') && !twitterUrl.includes('x.com')) {
      showError('batchError', 'Por favor, insira um link válido do Twitter/X.');
      return;
   }

   hideError('batchError');
   showLoading('batchLoading');
   hideResults('batchResults');

   try {
      const mockComments = [
         "Great post! Thanks for sharing.",
         "You people are disgusting and should not exist",
         "I love this community, everyone is so helpful!",
         "I hate all of you, you are worthless",
         "Amazing work, keep it up!",
         "This is terrible, you should be ashamed",
         "Fantastic content, very informative",
         "You are all idiots and don't deserve respect",
         "Thank you for this valuable information",
         "I can't stand people like you"
      ];

      const response = await fetch(`${API_BASE_URL}/predict/batch`, {
         method: 'POST',
         headers: {
            'Content-Type': 'application/json',
         },
         body: JSON.stringify({ comments: mockComments })
      });

      if (!response.ok) {
         throw new Error(`Erro na API: ${response.status} - ${response.statusText}`);
      }

      const result = await response.json();

      hideLoading('batchLoading');
      showBatchResults(result);

   } catch (error) {
      hideLoading('batchLoading');
      console.error('Erro:', error);

      if (error.name === 'TypeError' && error.message.includes('fetch')) {
         showError('batchError', 'Não foi possível conectar com o servidor. Verifique se a API está rodando.');
      } else {
         showError('batchError', `Erro ao processar os comentários: ${error.message}`);
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

function showBatchResults(result) {
   const hateComments = result.results.filter(r => r.is_hate_speech && !r.error);
   const safeComments = result.results.filter(r => !r.is_hate_speech && !r.error);
   const totalComments = result.results.filter(r => !r.error).length;

   document.getElementById('totalComments').textContent = totalComments;
   document.getElementById('hateComments').textContent = hateComments.length;
   document.getElementById('safeComments').textContent = safeComments.length;
   document.getElementById('hatePercentage').textContent =
      totalComments > 0 ? Math.round((hateComments.length / totalComments) * 100) + '%' : '0%';

   document.getElementById('hateCount').textContent = hateComments.length;
   document.getElementById('safeCount').textContent = safeComments.length;

   populateCommentList('hateCommentsList', hateComments, true);
   populateCommentList('safeCommentsList', safeComments, false);

   showResults('batchResults');
}

function populateCommentList(containerId, comments, isHate) {
   const container = document.getElementById(containerId);
   container.innerHTML = '';

   if (comments.length === 0) {
      container.innerHTML = '<p style="text-align: center; color: #666; font-style: italic;">Nenhum comentário encontrado nesta categoria.</p>';
      return;
   }

   comments.forEach(comment => {
      const commentDiv = document.createElement('div');
      commentDiv.className = `comment-item ${isHate ? 'hate-comment' : 'safe-comment'}`;

      commentDiv.innerHTML = `
                    <div class="comment-text">${comment.comment}</div>
                    <div class="comment-confidence">Confiança: ${comment.confidence}%</div>
                `;

      container.appendChild(commentDiv);
   });
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

function resetBatchAnalysis() {
   document.getElementById('twitterInput').value = '';
   hideResults('batchResults');
   hideError('batchError');
   hideLoading('batchLoading');
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