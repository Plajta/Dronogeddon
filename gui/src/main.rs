use gtk4 as gtk;
use gtk::prelude::*;
use std::
{
    process::Command,
    cel::Cell
};

use gtk::
{
    Application,
    ApplicationWindow,
    Button
};

//let const PROJECT_ROOT: &str = "./";

fn main() {
    let buttonUsed = Cell::new(false);

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
            if buttonUsed.get()
            {
                Command::new("python3")
                    .arg("src/video.py")
                    .spawn()
                    .expect("Script failed to run!");

                buttonUsed.set(!buttonUsed.get());
            } else
            {
                println!("You already started script!");
            }
        });

        window.set_child(Some(&button));

        window.show();
    });

    app.run();
}
