(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
  "use strict";
  
  var _ext = require("./utils/ext");
  
  var _ext2 = _interopRequireDefault(_ext);
  
  var _storage = require("./utils/storage");
  
  var _storage2 = _interopRequireDefault(_storage);
  
  var newsTitle;
  
  var bkg = chrome.extension.getBackgroundPage();

  var lastResult = null;
  
  function updateRecommendations(clickedWord, currentSentiment) {
    var updatedRecommendations = mockArticles.filter(function(article) {
      return article.keywords.some(function(keywordObj) {
        return keywordObj.word === clickedWord && keywordObj.sentiment !== currentSentiment;
      });
    });
  
    renderRecommendations(updatedRecommendations);
  }
  
  function renderRecommendations(recommendations, keywordsHTML = '') {
    var displayContainer = document.getElementById("display-container");
    var recommendationsHTML = "";
  
    var processedKeywordsHTML = processKeywordsForSentiment(keywordsHTML);
  
    recommendations.forEach(function(article) {
      // Process keywords and their sentiment
      var keywordsSentimentHTML = article.keywords.map(function(keywordObj) {
        var color = keywordObj.sentiment === 'positive' ? 'green' :
                    keywordObj.sentiment === 'negative' ? 'red' : 'blue';
        return `<span class="keyword" style="color: ${color};" data-word="${keywordObj.word}" data-sentiment="${keywordObj.sentiment}">${keywordObj.word}</span>`;
      }).join(', ');
      
      recommendationsHTML += `
        <div>
          <h3 class="title">${article.title}</h3>
          <p class="description">Source: ${article.source}</p>
          <div class="political-spectrum">
            <p>Political Leaning: ${createPoliticalSpectrum(article.M2_1_perspectives)}</p>
          </div>
          <p>Keywords and Sentiment: ${keywordsSentimentHTML}</p>
        </div>`;
    });
  
    displayContainer.innerHTML = processedKeywordsHTML + recommendationsHTML;
    bindKeywordButtonEvents();
  }
  
  function bindKeywordButtonEvents() {
    var keywordButtons = document.querySelectorAll('.keyword-button');
    keywordButtons.forEach(function(button) {
      button.addEventListener('click', function() {
        var word = this.getAttribute('data-word');
        var sentiment = this.getAttribute('data-sentiment');
        updateRecommendations(word, sentiment);
      });
    });
  }
  
  
  function createPoliticalSpectrum(leaning) {
    // const positions = { 'far left': 0, 'left': 25, 'center': 50, 'right': 75, 'far right': 100 };
    const positions = { '-2' : 0, '-1': 25, '0': 50, '1': 75, '2': 100 };
  
    let position = positions[leaning]; //|| 50; // Default to center if undefined
  
    return `
      <div class="spectrum-bar">
        <span class="spectrum-indicator" style="left: ${position}%"></span>
      </div>
      <div class="spectrum-labels">
        <span>Far Left</span>
        <span>Left</span>
        <span>Center</span>
        <span>Right</span>
        <span>Far Right</span>
      </div>
    `;
  }
  
  function processKeywordsForSentiment(keywordsHTML) {
    // Dummy function to process keywordsHTML. You will need to replace this logic
    // with actual processing based on how your keywordsHTML is structured.
    // This is just an example.
    return keywordsHTML.split(',').map(keyword => {
      // Example: Extract sentiment and apply color
      // Replace this with actual logic based on your data structure
      var sentiment = 'neutral'; // Dummy sentiment, replace with actual logic
      var color = sentiment === 'positive' ? 'green' :
                  sentiment === 'negative' ? 'red' : 'blue';
      return `<span style="color: ${color};">${keyword}</span>`;
    }).join(',');
  }
  
  function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
  
  var popup = document.getElementById("app");
  _storage2.default.get('color', function (resp) {
    var color = resp.color;
    if (color) {
      popup.style.backgroundColor = color;
    }
  });
  
  // render popup HTML template based on extracted data of webpage
  // 1. Title of webpage data
  // 2. Description of webpage data
  // 3. URL of webpage
  var template = function template(data) {
    var json = JSON.stringify(data);
    newsTitle = data.title;
    return "\n  <div class=\"site-description\">\n    <h3 class=\"title\">" + data.title + "</h3>\n    <p class=\"description\">" + data.description + "</p>\n    <a href=\"" + data.url + "\" target=\"_blank\" class=\"url\">" + data.url + "</a>\n  </div>\n  <div class=\"action-container\">\n    <button data-bookmark='" + json + "' id=\"predict-btn\" class=\"btn btn-primary\">" + 'Recommend' + "</button>\n  </div>\n  ";
  };
  
  // render the message in the body of display-container
  var renderMessage = function renderMessage(message) {
    var displayContainer = document.getElementById("display-container");
    displayContainer.innerHTML = "<p class='message'>" + message + "</p>";
  };
  
  function bindKeySentenceEvents() {
    var clickableKeywords = document.querySelectorAll('.clickable-keyword');
    clickableKeywords.forEach(function(keywordElement) {
      keywordElement.addEventListener('click', function() {
        var articleIndex = this.getAttribute('data-article-index');
        sendIndexToBackend(articleIndex);
      });
    });
  }

  function sendIndexToBackend(articleIndex) {
    var xhr = new XMLHttpRequest();
    // xhr.open("POST", "http://3.138.79.103:8000/analyze_this", true);
    xhr.open("POST", "http://3.138.79.103:8000/analyze_all", true);
    xhr.setRequestHeader("Content-type", "application/json");
  
    xhr.onreadystatechange = function() {
      if (xhr.readyState == 4 && xhr.status == 200) {
        var response = JSON.parse(xhr.responseText);
        displayAnalyzedInformation(response);
      }
    }
  
    var data = JSON.stringify({ article_idx: articleIndex });
    xhr.send(data);
  }

  var displayAnalyzedInformation = function(result) {
    var displayContainer = document.getElementById("display-container");
  
    // Start building the HTML string
    var html = `
      <button id="back-btn" class="btn btn-secondary">Back</button>
      
      <h4>User Article Title:</h4>
      <div class="article-cards">
        <div class="article-card">
        <p>${result.user_article.title}</p>
        <p>Keyword: <strong>${result.user_article.keyword}</strong> (Sentiment: ${result.user_article.sentiment})</p>
        </div>
      </div>

      <h4>Queried Articles:</h4>
      <div class="article-cards">
    `;
  
    // Loop through queried_articles and create a card for each article
    result.queried_articles.forEach(function(article, index) {
      html += `
        <div class="article-card">
          <a href="${article.url}" target="_blank"><h4>${article.title}</h4></a>
          <p>Keyword: <strong>${article.keyword}</strong> (Sentiment: ${article.sentiment})</p>
        </div>
      `;
    });
  
    // Close the article-cards div and display the HTML
    html += `</div>`;
    displayContainer.innerHTML = html;

    document.getElementById("back-btn").addEventListener("click", function() {
      renderArticle(lastResult); // Call renderArticle with the stored result
    });
  
    bindKeySentenceEvents();
  };
  
  

  var renderArticle = function(result) {
    lastResult = result;
    var displayContainer = document.getElementById("display-container");
    
    var userArticleKeySentenceBolded = result.user_article.key_sentence.replace(
      new RegExp(result.user_article.keyword, 'gi'), 
      `<strong>$&</strong>`
    );
    // Start building the HTML string
    var html = `
      <h4>Article Title:</h4>
      <div class="article-cards">
        <div class="article-card">
          <h3><strong>${result.user_article.title}</strong></h3>
          <div class="political-spectrum">
            <p>Political Leaning: ${createPoliticalSpectrum(result.user_article.M2_1_perspectives)}</p>
          </div>
          <p>${userArticleKeySentenceBolded}</p>
        </div>
    
      <h4>Recommended Articles:</h4>
      <div class="article-cards">
    `;
    
    // Loop through queried_articles and create a card for each article
    // Loop through queried_articles and create a card for each article
    result.queried_articles.forEach(function(article, index) {
      var keySentenceWithClickableKeyword = article.key_sentence.replace(
        new RegExp(article.keyword, 'gi'), // 'gi' for case-insensitive match
        `<strong><span class="clickable-keyword" style="text-decoration: underline; color: blue;" data-article-index="${index}">$&</span></strong>`
      );
      html += `
        <div class="article-card">
          <a href="${article.url}" target="_blank"><h3 class="title">${article.title}</h3></a>
          <div class="political-spectrum">
            <p>Political Leaning: ${createPoliticalSpectrum(article.M2_1_perspectives)}</p>
          </div>
          <p>${keySentenceWithClickableKeyword}</p>
        </div>
      `;
    });

    
    // Close the article-cards div and display the HTML
    html += `</div>`;
    displayContainer.innerHTML = html;
  
    bindKeySentenceEvents();
  };
  
  // render the data if available
  var renderBookmark = function renderBookmark(data) {
    var displayContainer = document.getElementById("display-container");
    if (data) {
      var tmpl = template(data);
      displayContainer.innerHTML = tmpl;
      // console.log(displayContainer);
    } else {
      renderMessage("Sorry, could not extract this page's title and URL");
    }
  };
  
  // This function displays the content of the extension (process-page)
  // renderBookmark is called here to display the message onto the extension
  _ext2.default.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    var activeTab = tabs[0];
    console.log(activeTab.title); //title
    chrome.tabs.sendMessage(activeTab.id, { action: 'process-page' }, renderBookmark);
  });
  
  
  // Button
  popup.addEventListener("click", function (e) {
    if (e.target && e.target.matches("#predict-btn")) {
      e.preventDefault();
      var data = e.target.getAttribute("data-bookmark");
      // cant seem to change 'perform-save'
      _ext2.default.runtime.sendMessage({ action: "perform-save", data: data }, function (response) {
  
        if (response && response.action === "saved") {
  
          var data = JSON.stringify({text: [newsTitle]});
          var xhr = new XMLHttpRequest();
          console.log('success'); 
          // xhr.open("POST", "http://3.138.79.103:8000//analyze_this", true);
          xhr.open("POST", "http://3.138.79.103:8000/predict", true);
          // xhr.open("POST", "http://108.233.179.22:8080/predict", true);
          xhr.setRequestHeader("Content-type", "application/json");
          xhr.onreadystatechange = function() {
            console.log("Ready State:", xhr.readyState);
            if (xhr.readyState == 4 && xhr.status == 200) {
              // Parse the JSON response
              console.log('success'); 
              var obj = JSON.parse(xhr.responseText);
              console.log(obj); 
  
              renderArticle(obj);
              
            }
          }
          xhr.send(data);
  
        } else {
          renderMessage("Sorry, there was an error while predicting your article.");
        }
      });
    }
  });
  
  var optionsLink = document.querySelector(".js-options");
  optionsLink.addEventListener("click", function (e) {
    e.preventDefault();
    _ext2.default.tabs.create({ 'url': _ext2.default.extension.getURL('options.html') });
  });
  
  },{"./utils/ext":2,"./utils/storage":3}],2:[function(require,module,exports){
  'use strict';
  
  var apis = ['alarms', 'bookmarks', 'browserAction', 'commands', 'contextMenus', 'cookies', 'downloads', 'events', 'extension', 'extensionTypes', 'history', 'i18n', 'idle', 'notifications', 'pageAction', 'runtime', 'storage', 'tabs', 'webNavigation', 'webRequest', 'windows'];
  
  function Extension() {
    var _this = this;
  
    apis.forEach(function (api) {
  
      _this[api] = null;
  
      try {
        if (chrome[api]) {
          _this[api] = chrome[api];
        }
      } catch (e) {}
  
      try {
        if (window[api]) {
          _this[api] = window[api];
        }
      } catch (e) {}
  
      try {
        if (browser[api]) {
          _this[api] = browser[api];
        }
      } catch (e) {}
      try {
        _this.api = browser.extension[api];
      } catch (e) {}
    });
  
    try {
      if (browser && browser.runtime) {
        this.runtime = browser.runtime;
      }
    } catch (e) {}
  
    try {
      if (browser && browser.browserAction) {
        this.browserAction = browser.browserAction;
      }
    } catch (e) {}
  }
  
  module.exports = new Extension();
  
  },{}],3:[function(require,module,exports){
  "use strict";
  
  var _ext = require("./ext");
  
  var _ext2 = _interopRequireDefault(_ext);
  
  function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }
  
  module.exports = _ext2.default.storage.sync ? _ext2.default.storage.sync : _ext2.default.storage.local;
  
  },{"./ext":2}]},{},[1])
  // # sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJzcmMvc2NyaXB0cy9wb3B1cC5qcyIsInNyYy9zY3JpcHRzL3V0aWxzL2V4dC5qcyIsInNyYy9zY3JpcHRzL3V0aWxzL3N0b3JhZ2UuanMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7OztBQ0FBOzs7O0FBQ0E7Ozs7OztBQUVBLElBQUksUUFBUSxTQUFTLGNBQVQsQ0FBd0IsS0FBeEIsQ0FBWjtBQUNBLGtCQUFRLEdBQVIsQ0FBWSxPQUFaLEVBQXFCLFVBQVMsSUFBVCxFQUFlO0FBQ2xDLE1BQUksUUFBUSxLQUFLLEtBQWpCO0FBQ0EsTUFBRyxLQUFILEVBQVU7QUFDUixVQUFNLEtBQU4sQ0FBWSxlQUFaLEdBQThCLEtBQTlCO0FBQ0Q7QUFDRixDQUxEOztBQU9BLElBQUksV0FBVyxTQUFYLFFBQVcsQ0FBQyxJQUFELEVBQVU7QUFDdkIsTUFBSSxPQUFPLEtBQUssU0FBTCxDQUFlLElBQWYsQ0FBWDtBQUNBLDRFQUVzQixLQUFLLEtBRjNCLDRDQUcyQixLQUFLLFdBSGhDLDRCQUlhLEtBQUssR0FKbEIsMkNBSXNELEtBQUssR0FKM0QsdUZBTzJCLElBUDNCO0FBVUQsQ0FaRDtBQWFBLElBQUksZ0JBQWdCLFNBQWhCLGFBQWdCLENBQUMsT0FBRCxFQUFhO0FBQy9CLE1BQUksbUJBQW1CLFNBQVMsY0FBVCxDQUF3QixtQkFBeEIsQ0FBdkI7QUFDQSxtQkFBaUIsU0FBakIsMkJBQW1ELE9BQW5EO0FBQ0QsQ0FIRDs7QUFLQSxJQUFJLGlCQUFpQixTQUFqQixjQUFpQixDQUFDLElBQUQsRUFBVTtBQUM3QixNQUFJLG1CQUFtQixTQUFTLGNBQVQsQ0FBd0IsbUJBQXhCLENBQXZCO0FBQ0EsTUFBRyxJQUFILEVBQVM7QUFDUCxRQUFJLE9BQU8sU0FBUyxJQUFULENBQVg7QUFDQSxxQkFBaUIsU0FBakIsR0FBNkIsSUFBN0I7QUFDRCxHQUhELE1BR087QUFDTCxrQkFBYyxvREFBZDtB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O0FBQ3BDLFdBQUssYUFBTCxHQUFxQixRQUFRLGFBQTdCO0FBQ0Q7QUFDRixHQUpELENBSUUsT0FBTyxDQUFQLEVBQVUsQ0FBRTtBQUVmOztBQUVELE9BQU8sT0FBUCxHQUFpQixJQUFJLFNBQUosRUFBakI7Ozs7O0FDbkVBOzs7Ozs7QUFFQSxPQUFPLE9BQVAsR0FBa0IsY0FBSSxPQUFKLENBQVksSUFBWixHQUFtQixjQUFJLE9BQUosQ0FBWSxJQUEvQixHQUFzQyxjQUFJLE9BQUosQ0FBWSxLQUFwRSIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCJpbXBvcnQgZXh0IGZyb20gXCIuL3V0aWxzL2V4dFwiO1xuaW1wb3J0IHN0b3JhZ2UgZnJvbSBcIi4vdXRpbHMvc3RvcmFnZVwiO1xuXG52YXIgcG9wdXAgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChcImFwcFwiKTtcbnN0b3JhZ2UuZ2V0KCdjb2xvcicsIGZ1bmN0aW9uKHJlc3ApIHtcbiAgdmFyIGNvbG9yID0gcmVzcC5jb2xvcjtcbiAgaWYoY29sb3IpIHtcbiAgICBwb3B1cC5zdHlsZS5iYWNrZ3JvdW5kQ29sb3IgPSBjb2xvclxuICB9XG59KTtcblxudmFyIHRlbXBsYXRlID0gKGRhdGEpID0+IHtcbiAgdmFyIGpzb24gPSBKU09OLnN0cmluZ2lmeShkYXRhKTtcbiAgcmV0dXJuIChgXG4gIDxkaXYgY2xhc3M9XCJzaXRlLWRlc2NyaXB0aW9uXCI+XG4gICAgPGgzIGNsYXNzPVwidGl0bGVcIj4ke2RhdGEudGl0bGV9PC9oMz5cbiAgICA8cCBjbGFzcz1cImRlc2NyaXB0aW9uXCI+JHtkYXRhLmRlc2NyaXB0aW9ufTwvcD5cbiAgICA8YSBocmVmPVwiJHtkYXRhLnVybH1cIiB0YXJnZXQ9XCJfYmxhbmtcIiBjbGFzcz1cInVybFwiPiR7ZGF0YS51cmx9PC9hPlxuICA8L2Rpdj5cbiAgPGRpdiBjbGFzcz1cImFjdGlvbi1jb250YWluZXJcIj5cbiAgICA8YnV0dG9uIGRhdGEtYm9va21hcms9JyR7anNvbn0nIGlkPVwic2F2ZS1idG5cIiBjbGFzcz1cImJ0biBidG4tcHJpbWFyeVwiPlNhdmU8L2J1dHRvbj5cbiAgPC9kaXY+XG4gIGApO1xufVxudmFyIHJlbmRlck1lc3NhZ2UgPSAobWVzc2FnZSkgPT4ge1xuICB2YXIgZGlzcGxheUNvbnRhaW5lciA9IGRvY3VtZW50LmdldEVsZW1lbnRCeUlkKFwiZGlzcGxheS1jb250YWluZXJcIik7XG4gIGRpc3BsYXlDb250YWluZXIuaW5uZXJIVE1MID0gYDxwIGNsYXNzPSdtZXNzYWdlJz4ke21lc3NhZ2V9PC9wPmA7XG59XG5cbnZhciByZW5kZXJCb29rbWFyayA9IChkYXRhKSA9PiB7XG4gIHZhciBkaXNwbGF5Q29udGFpbmVyID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoXCJkaXNwbGF5LWNvbnRhaW5lclwiKVxuICBpZihkYXRhKSB7XG4gICAgdmFyIHRtcGwgPSB0ZW1wbGF0ZShkYXRhKTtcbiAgICBkaXNwbGF5Q29udGFpbmVyLmlubmVySFRNTCA9IHRtcGw7ICBcbiAgfSBlbHNlIHtcbiAgICByZW5kZXJNZXNzYWdlKFwiU29ycnksIGNvdWxkIG5vdCBleHRyYWN0IHRoaXMgcGFnZSdzIHRpdGxlIGFuZCBVUkxcIilcbiAgfVxufVxuXG5leHQudGFicy5xdWVyeSh7YWN0aXZlOiB0cnVlLCBjdXJyZW50V2luZG93OiB0cnVlfSwgZnVuY3Rpb24odGFicykge1xuICB2YXIgYWN0aXZlVGFiID0gdGFic1swXTtcbiAgY2hyb21lLnRhYnMuc2VuZE1lc3NhZ2UoYWN0aXZlVGFiLmlkLCB7IGFjdGlvbjogJ3Byb2Nlc3MtcGFnZScgfSwgcmVuZGVyQm9va21hcmspO1xufSk7XG5cbnBvcHVwLmFkZEV2ZW50TGlzdGVuZXIoXCJjbGlja1wiLCBmdW5jdGlvbihlKSB7XG4gIGlmKGUudGFyZ2V0ICYmIGUudGFyZ2V0Lm1hdGNoZXMoXCIjc2F2ZS1idG5cIikpIHtcbiAgICBlLnByZXZlbnREZWZhdWx0KCk7XG4gICAgdmFyIGRhdGEgPSBlLnRhcmdldC5nZXRBdHRyaWJ1dGUoXCJkYXRhLWJvb2ttYXJrXCIpO1xuICAgIGV4dC5ydW50aW1lLnNlbmRNZXNzYWdlKHsgYWN0aW9uOiBcInBlcmZvcm0tc2F2ZVwiLCBkYXRhOiBkYXRhIH0sIGZ1bmN0aW9uKHJlc3BvbnNlKSB7XG4gICAgICBpZihyZXNwb25zZSAmJiByZXNwb25zZS5hY3Rpb24gPT09IFwic2F2ZWRcIikge1xuICAgICAgICByZW5kZXJNZXNzYWdlKFwiWW91ciBib29rbWFyayB3YXMgc2F2ZWQgc3VjY2Vzc2Z1bGx5IVwiKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJlbmRlck1lc3NhZ2UoXCJTb3JyeSwgdGhlcmUgd2FzIGFuIGVycm9yIHdoaWxlIHNhdmluZyB5b3VyIGJvb2ttYXJrLlwiKTtcbiAgICAgIH1cbiAgICB9KVxuICB9XG59KTtcblxudmFyIG9wdGlvbnNMaW5rID0gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihcIi5qcy1vcHRpb25zXCIpO1xub3B0aW9uc0xpbmsuYWRkRXZlbnRMaXN0ZW5lcihcImNsaWNrXCIsIGZ1bmN0aW9uKGUpIHtcbiAgZS5wcmV2ZW50RGVmYXVsdCgpO1xuICBleHQudGFicy5jcmVhdGUoeyd1cmwnOiBleHQuZXh0ZW5zaW9uLmdldFVSTCgnb3B0aW9ucy5odG1sJyl9KTtcbn0pXG4iLCJjb25zdCBhcGlzID0gW1xuICAnYWxhcm1zJyxcbiAgJ2Jvb2ttYXJrcycsXG4gICdicm93c2VyQWN0aW9uJyxcbiAgJ2NvbW1hbmRzJyxcbiAgJ2NvbnRleHRNZW51cycsXG4gICdjb29raWVzJyxcbiAgJ2Rvd25sb2FkcycsXG4gICdldmVudHMnLFxuICAnZXh0ZW5zaW9uJyxcbiAgJ2V4dGVuc2lvblR5cGVzJyxcbiAgJ2hpc3RvcnknLFxuICAnaTE4bicsXG4gICdpZGxlJyxcbiAgJ25vdGlmaWNhdGlvbnMnLFxuICAncGFnZUFjdGlvbicsXG4gICdydW50aW1lJyxcbiAgJ3N0b3JhZ2UnLFxuICAndGFicycsXG4gICd3ZWJOYXZpZ2F0aW9uJyxcbiAgJ3dlYlJlcXVlc3QnLFxuICAnd2luZG93cycsXG5dXG5cbmZ1bmN0aW9uIEV4dGVuc2lvbiAoKSB7XG4gIGNvbnN0IF90aGlzID0gdGhpc1xuXG4gIGFwaXMuZm9yRWFjaChmdW5jdGlvbiAoYXBpKSB7XG5cbiAgICBfdGhpc1thcGldID0gbnVsbFxuXG4gICAgdHJ5IHtcbiAgICAgIGlmIChjaHJvbWVbYXBpXSkge1xuICAgICAgICBfdGhpc1thcGldID0gY2hyb21lW2FwaV1cbiAgICAgIH1cbiAgICB9IGNhdGNoIChlKSB7fVxuXG4gICAgdHJ5IHtcbiAgICAgIGlmICh3aW5kb3dbYXBpXSkge1xuICAgICAgICBfdGhpc1thcGldID0gd2luZG93W2FwaV1cbiAgICAgIH1cbiAgICB9IGNhdGNoIChlKSB7fVxuXG4gICAgdHJ5IHtcbiAgICAgIGlmIChicm93c2VyW2FwaV0pIHtcbiAgICAgICAgX3RoaXNbYXBpXSA9IGJyb3dzZXJbYXBpXVxuICAgICAgfVxuICAgIH0gY2F0Y2ggKGUpIHt9XG4gICAgdHJ5IHtcbiAgICAgIF90aGlzLmFwaSA9IGJyb3dzZXIuZXh0ZW5zaW9uW2FwaV1cbiAgICB9IGNhdGNoIChlKSB7fVxuICB9KVxuXG4gIHRyeSB7XG4gICAgaWYgKGJyb3dzZXIgJiYgYnJvd3Nlci5ydW50aW1lKSB7XG4gICAgICB0aGlzLnJ1bnRpbWUgPSBicm93c2VyLnJ1bnRpbWVcbiAgICB9XG4gIH0gY2F0Y2ggKGUpIHt9XG5cbiAgdHJ5IHtcbiAgICBpZiAoYnJvd3NlciAmJiBicm93c2VyLmJyb3dzZXJBY3Rpb24pIHtcbiAgICAgIHRoaXMuYnJvd3NlckFjdGlvbiA9IGJyb3dzZXIuYnJvd3NlckFjdGlvblxuICAgIH1cbiAgfSBjYXRjaCAoZSkge31cblxufVxuXG5tb2R1bGUuZXhwb3J0cyA9IG5ldyBFeHRlbnNpb24oKTsiLCJpbXBvcnQgZXh0IGZyb20gXCIuL2V4dFwiO1xuXG5tb2R1bGUuZXhwb3J0cyA9IChleHQuc3RvcmFnZS5zeW5jID8gZXh0LnN0b3JhZ2Uuc3luYyA6IGV4dC5zdG9yYWdlLmxvY2FsKTsiXX0=
  
  