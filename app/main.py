from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.routers import iris, titanic, wine


APP_VERSION = '2.0.0'


def create_application() -> FastAPI:
    """Создает и настраивает экземпляр FastAPI-приложения."""

    settings = get_settings()

    application = FastAPI(
        title='ML Predictor API',
        description='API для предсказаний: ирисы, вино и выживание на Титанике',
        version=APP_VERSION,
        docs_url='/api/docs',
        redoc_url='/api/redoc',
    )

    application.mount(
        '/static',
        StaticFiles(directory=settings.STATIC_DIR),
        name='static',
    )

    application.include_router(iris.router)
    application.include_router(wine.router)
    application.include_router(titanic.router)

    return application


app = create_application()
templates = Jinja2Templates(directory=get_settings().TEMPLATES_DIR)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Возвращает единый формат ошибки валидации входных данных."""

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            'message': 'Некорректные входные данные',
            'detail': exc.errors(),
        },
    )


@app.get('/', response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """Отдает главную HTML-страницу приложения."""

    return templates.TemplateResponse(
        request=request,
        name='index.html',
        context={},
    )


@app.get('/health')
async def health_check() -> dict[str, str]:
    """Возвращает состояние приложения для быстрой проверки доступности."""

    return {'status': 'healthy', 'version': APP_VERSION}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('app.main:app', host='0.0.0.0', port=8000, reload=True)
