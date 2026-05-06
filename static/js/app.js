const currentState = {
    model: 'iris'
};

const formFieldIds = {
    iris: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    wine: [
        'alcohol',
        'malic_acid',
        'ash',
        'alcalinity_of_ash',
        'magnesium',
        'total_phenols',
        'flavanoids',
        'nonflavanoid_phenols',
        'proanthocyanins',
        'color_intensity',
        'hue',
        'od280_od315',
        'proline'
    ],
    titanic: ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
};

const integerFields = new Set(['pclass', 'sex', 'sibsp', 'parch', 'embarked']);

function switchModel(model, button = null) {
    currentState.model = model;

    document.querySelectorAll('.model-btn').forEach((btn) => {
        btn.classList.remove('active');
    });

    if (button) {
        button.classList.add('active');
    }

    document.querySelectorAll('.model-section').forEach((section) => {
        section.classList.remove('active');
    });

    document.getElementById(`${model}-section`).classList.add('active');
    hideResults();
    hideError();
}

function collectFeatures(model) {
    return formFieldIds[model].reduce((features, fieldId) => {
        const value = document.getElementById(fieldId).value;
        features[fieldId] = integerFields.has(fieldId) ? parseInt(value, 10) : parseFloat(value);
        return features;
    }, {});
}

function registerForm(model) {
    const form = document.getElementById(`${model}-form`);

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        await predict(model, collectFeatures(model));
    });
}

async function predict(model, features) {
    const button = document.querySelector(`#${model}-form .predict-btn`);
    const buttonText = button.querySelector('.btn-text');
    const buttonLoader = button.querySelector('.btn-loader');

    setLoading(button, buttonText, buttonLoader, true);
    hideError();

    try {
        const response = await fetch(`/api/${model}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(features)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(formatApiError(data));
        }

        displayResults(data, model);
    } catch (error) {
        showError(error.message || 'Не удалось получить предсказание');
    } finally {
        setLoading(button, buttonText, buttonLoader, false);
    }
}

function setLoading(button, buttonText, buttonLoader, isLoading) {
    button.disabled = isLoading;
    buttonText.classList.toggle('hidden', isLoading);
    buttonLoader.classList.toggle('hidden', !isLoading);
}

function formatApiError(data) {
    if (data.message) {
        return data.message;
    }

    if (typeof data.detail === 'string') {
        return data.detail;
    }

    if (data.detail?.message) {
        return data.detail.message;
    }

    if (Array.isArray(data.detail) && data.detail.length > 0) {
        return data.detail
            .map((item) => `${item.loc?.join('.') || 'field'}: ${item.msg}`)
            .join('\n');
    }

    return 'Ошибка API';
}

function showError(message) {
    const container = getErrorContainer();
    container.textContent = message;
    container.classList.remove('hidden');
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    const container = getErrorContainer();
    container.classList.add('hidden');
    container.textContent = '';
}

function getErrorContainer() {
    let container = document.getElementById('error-container');

    if (!container) {
        container = document.createElement('div');
        container.id = 'error-container';
        container.className = 'error-container hidden';
        document.querySelector('.container').appendChild(container);
    }

    return container;
}

function hideResults() {
    document.getElementById('result-container').classList.add('hidden');
}

function displayResults(data, model) {
    const container = document.getElementById('result-container');
    const card = document.getElementById('result-card');
    const classElement = document.getElementById('predicted-class');
    const confidenceElement = document.getElementById('confidence');
    const probabilitiesElement = document.getElementById('probabilities');
    const title = document.getElementById('result-title');
    const survivalMeter = document.getElementById('survival-meter');

    if (model === 'titanic') {
        const survivalPercentage = data.survival_percentage;

        card.classList.add('titanic-result');
        title.textContent = '🚢 Анализ выживаемости';
        survivalMeter.classList.remove('hidden');
        document.getElementById('meter-fill').style.width = `${survivalPercentage}%`;
        document.getElementById('meter-value').textContent = `${survivalPercentage.toFixed(1)}%`;
        classElement.textContent = survivalPercentage > 50 ? 'Выжил бы ✅' : 'Не выжил ❌';
    } else {
        card.classList.remove('titanic-result');
        title.textContent = 'Результат предсказания';
        survivalMeter.classList.add('hidden');
        classElement.textContent = data.predicted_class;
    }

    confidenceElement.textContent = `Уверенность: ${(data.confidence * 100).toFixed(1)}%`;
    renderProbabilities(probabilitiesElement, data.probabilities, model);

    container.classList.remove('hidden');
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderProbabilities(container, probabilities, model) {
    container.innerHTML = '';

    Object.entries(probabilities)
        .sort(([, first], [, second]) => second - first)
        .forEach(([className, probability]) => {
            const bar = document.createElement('div');
            bar.className = 'prob-bar';
            const fillClass = model === 'titanic' ? 'titanic-fill' : '';

            bar.innerHTML = `
                <span class='prob-label'>${className}</span>
                <div class='prob-track'>
                    <div class='prob-fill ${fillClass}' style='width: 0%'>
                        <span class='prob-value'>${(probability * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;

            container.appendChild(bar);

            setTimeout(() => {
                bar.querySelector('.prob-fill').style.width = `${probability * 100}%`;
            }, 100);
        });
}

Object.keys(formFieldIds).forEach(registerForm);
