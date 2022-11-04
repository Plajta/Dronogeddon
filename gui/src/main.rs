use gtk4 as gtk;
use gtk::prelude::*;
use gtk::
{
    Application,
    ApplicationWindow,
    Button
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

        let button = Button::with_label("Click me!");

        button.connect_clicked(|_|
        {
            println!("Clicked!");
        });

        window.set_child(Some(&button));

        window.show();
    });

    app.run();
}
