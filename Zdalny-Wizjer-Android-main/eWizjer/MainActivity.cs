using Android.App;
using Android.Content;
using Android.Graphics;
using Android.OS;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using AndroidX.AppCompat.App;
using AndroidX.AppCompat.Widget;
using eWizjer;
using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using Wizjer;
//using eWizjer.Resources;


namespace eWizjer
{

    [Activity(Label = "@string/app_name", Theme = "@style/Theme.AppCompat.Light.NoActionBar")]
    public class MainActivity : AppCompatActivity
    {

        ImageView thisImageView;
        private System.Timers.Timer timer = new System.Timers.Timer();
        private System.Timers.Timer timer_init = new System.Timers.Timer();

        public static ClientWebSocket cl = new ClientWebSocket();
        private Uri uri = new Uri("ws://192.168.0.108:5050/");
        private byte preson_list;
        //support values
        private int test_list_cnt = 0;
        private bool allow_read_file = false;
        string person_list_str;

        string[] person_table;

        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);
            Xamarin.Essentials.Platform.Init(this, savedInstanceState);
            // Set our view from the "main" layout resource
            SetContentView(Resource.Layout.activity_main);
            thisImageView = (ImageView)FindViewById(Resource.Id.imageView1);
            AndroidX.AppCompat.Widget.Toolbar toolbar = FindViewById<AndroidX.AppCompat.Widget.Toolbar>(Resource.Id.toolbarMainPage);
            SetSupportActionBar(toolbar);

            Button button = FindViewById<Button>(Resource.Id.OpenButton);
            button.Click += delegate
            {
                EditText et = new EditText(this);
                et.InputType = Android.Text.InputTypes.TextVariationPassword |
                          Android.Text.InputTypes.ClassText;
                Android.App.AlertDialog.Builder dialog = new Android.App.AlertDialog.Builder(this);
                Android.App.AlertDialog alert = dialog.Create();
                alert.SetTitle("Enter your password");
                alert.SetView(et); // <----
                alert.SetButton("Cancel", (c, ev) =>
                {
                    alert.Dispose();
                });
                alert.SetButton2("Open", (c, ev) =>
                {
                    string password = et.Text.ToString();
                    if (password == "Admin")
                    {
                        Toast.MakeText(this, "Doors open", ToastLength.Short).Show();
                        alert.Dispose();
                    }
                    else
                    {
                        alert.Dispose();
                        Toast.MakeText(this, "incorrect password", ToastLength.Short).Show();
                    }
                });
                alert.Show();
            };


           
            timer_init.Interval = 1000;
            timer_init.Elapsed += Timer_init_Elapsed;
            timer_init.AutoReset = false;


            timer.Interval = 100;
            timer.Elapsed += Timer_Elapsed;
            timer.AutoReset = true;
            //wyswietlij text ze sie polaczyl


            //uruchom zadanie do odbioru paczek z serwera => tam bedzie petla
            //Task.Run(() => ReceiveMsgAsync());

            //wyswietl komuikat na ekranie
            

            //wystaryj timer do odbioru i wyswietlania obrazu
            timer_init.Start();

        }

        private void Timer_init_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            timer_init.Stop();
            System.Threading.Tasks.Task.Run(async () => await cl.ConnectAsync(uri, System.Threading.CancellationToken.None));

            //czekaj az client sie polaczy
            while (cl.State != WebSocketState.Open)
            {

            }
            Task.Run(() => ReceiveMsgAsync());
            timer.Start();

        }

        public override bool OnCreateOptionsMenu(IMenu menu)
        {


            this.MenuInflater.Inflate(Resource.Menu.toolbarListMenu, menu);
            return true;
        }

        
        public override bool OnOptionsItemSelected(IMenuItem item)
        {
            int id = item.ItemId;
            if (id == Resource.Id.addPicture)
            {
                // Toast.MakeText(this, "Add Piture", ToastLength.Short).Show();
                //Get_person_list();
                // this.StartActivity(typeof(PersonListActivity));
                person_list_str = " a";
              
                    if (person_list_str != null) {
                        Intent addPictureActivity = new Intent(this, typeof(PersonListActivity));
                        addPictureActivity.PutExtra("person_list", person_list_str);
                    StartActivity(addPictureActivity);

                }

                
                return true;
            }
            if (id == Resource.Id.pearsonList)
            {
                Toast.MakeText(this, "The Raspberry Pi has restarted", ToastLength.Short).Show();
                return true;
            }
            if (id == Resource.Id.logout)
            {
                Android.App.AlertDialog.Builder dialog = new Android.App.AlertDialog.Builder(this);
                //Android.App.AlertDialog alert = dialog.Create();
                dialog.SetTitle("Confirm Logout");
                dialog.SetMessage("You definitely want to log out?");
                dialog.SetPositiveButton("Yes", (senderAlert, args) => {
                    Toast.MakeText(this, "Logout", ToastLength.Short).Show();
                    StartActivity(typeof(LoginActivity));
                });
                dialog.SetNegativeButton("No", (senderAlert, args) => {
                    dialog.Dispose();
                });
                
                
                Dialog diag = dialog.Create();
                diag.Show();
                
                return true;
            }
            return true;
        }

        public override void OnRequestPermissionsResult(int requestCode, string[] permissions, [GeneratedEnum] Android.Content.PM.Permission[] grantResults)
        {
            Xamarin.Essentials.Platform.OnRequestPermissionsResult(requestCode, permissions, grantResults);

            base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        }

        private void Timer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            this.test_list_cnt++;
            
            //if (this.test_list_cnt == 1)
            //    timer.Stop();
            //gdy można odczyac plik wyswietl obraz
            if (this.allow_read_file == true)
                DisplyPIC();

            //wyslij GET do serwera
            Task.Run(() => GetPIC());


        }

        private async void GetPIC()
        {
            string ss = "Get_picture";

            byte[] bb = System.Text.ASCIIEncoding.ASCII.GetBytes(ss);

            ArraySegment<byte> bbx = new ArraySegment<byte>(bb);
            this.allow_read_file = false;

            //wyslanie GET do serwera
            await cl.SendAsync(bbx, WebSocketMessageType.Binary, true, System.Threading.CancellationToken.None);

        }
        private async void Get_person_list()
        {
            string ss = "Get_person_list";

            byte[] bb = System.Text.ASCIIEncoding.ASCII.GetBytes(ss);

            ArraySegment<byte> bbx = new ArraySegment<byte>(bb);
            this.allow_read_file = false;

            //wyslanie GET do serwera
            await cl.SendAsync(bbx, WebSocketMessageType.Binary, true, System.Threading.CancellationToken.None);

        }
        private void DisplyPIC()
        {
            try
            {
                string t_path = System.IO.Path.Combine(Xamarin.Essentials.FileSystem.AppDataDirectory, "image.jpg");
                //tu sie bawilem z odczytem wlasiwosci plik
                System.IO.FileInfo fi = new System.IO.FileInfo(t_path);
                string nlen = "-";
                try
                {
                    nlen = fi.Length.ToString();
                    nlen = fi.Name + ", " + nlen + ", " + fi.LastWriteTime.ToString("HH:mm:ss");
                }
                catch { }
                //znalezienie ImageView1 z cantent_main.xml
                Android.Widget.ImageView imview = FindViewById<Android.Widget.ImageView>(Resource.Id.imageView1);
                //odczytanie bitmapy z pliku
                Bitmap myBitmap = BitmapFactory.DecodeFile(t_path);
                //podstawienie bitmapy do ImageView1 w glownym watku
                imview.Post(() => imview.SetImageBitmap(myBitmap));
            }
            catch (Exception e)
            {
                string ss = e.Message;
            }    
        }

        private async Task ReceiveMsgAsync()
        {
            try
            {
                //petla do odczytu paczki z serwera
                while (cl.State == WebSocketState.Open)
                {
                    try
                    {
                        byte[] InicByteArray = new byte[140000];
                        ArraySegment<byte> RecvByteArray = new ArraySegment<byte>(InicByteArray);
                        //czekaj az bedzie odczyt z serwera
                        WebSocketReceiveResult result = await cl.ReceiveAsync(RecvByteArray, System.Threading.CancellationToken.None);
                        int resultlen = result.Count;

                        byte[] dataByteArray = new byte[resultlen];
                        byte[] command = new byte[100];

                        System.Array.Copy(RecvByteArray.Array, 0, command, 0, 100);
                        System.Array.Copy(RecvByteArray.Array, 100, dataByteArray, 0, result.Count);

                        string command_decode = System.Text.Encoding.Default.GetString(command).Trim();
                        //zapisz do pliku to co odczytales
                        if (command_decode == "Get_picture")
                        {
                            string t_path = System.IO.Path.Combine(Xamarin.Essentials.FileSystem.AppDataDirectory, "image.jpg");
                            System.IO.File.WriteAllBytes(t_path, dataByteArray);
                            
                        }
                        if (command_decode == "Get_person_list")
                        {
                            person_list_str = Encoding.UTF8.GetString(dataByteArray);
                            person_table = person_list_str.Split(' ');
                        }                 
                    }
                    catch (Exception e)
                    {
                    }
                    this.allow_read_file = true;
                    
                }
            }
            catch (Exception exc)
            {
                string msg = exc.Message;
            }
        }


    }

}