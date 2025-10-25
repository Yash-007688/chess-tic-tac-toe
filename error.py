from flask import render_template

def register_error_handlers(app):
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template('500.html'), 500

    @app.errorhandler(Exception)
    def handle_exception(error):
        # Log the error or handle it
        return render_template('error.html', error=str(error)), 500