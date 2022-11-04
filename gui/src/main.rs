use gtk4 as gtk;
use gtk::prelude::*;
use gtk::
{
    Application,
    ApplicationWindow
};

fn main() {
    let app = Application::builder()
        .application_id("com.plajtacorp.dronogeddon")
        .build();

    app.connect_activate(|app|
    {
        let window = ApplicationWindow::builder()
            .application(app)
            .default_width(320)
            .default_height(200)
            .title("wannabe Control Panel")
            .build();

        window.show();
    );

    app.run();
}
