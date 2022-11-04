use gtk4 as gtk;
use gtk::prelude::*;
use std::process::Command;

use gtk::
{
    Application,
    ApplicationWindow,
    Button
};

//let const PROJECT_ROOT: &str = "./";
let mut buttonUsed: bool = false;

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
            if buttonUsed
            {
                Command::new("python3")
                    .arg("src/video.py")
                    .spawn()
                    .expect("Script failed to run!");

                buttonUsed = !buttonUsed;
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
