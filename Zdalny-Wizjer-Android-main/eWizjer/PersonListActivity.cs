using Android.App;
using Android.Content;
using Android.OS;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System;
using Plugin.Media;
using System.Drawing;
using Android.Graphics;
using Android;
using System.Net.WebSockets;

namespace eWizjer
{
    [Activity(Label = "PersonListActivity")]
    [Obsolete]
    public class PersonListActivity : ListActivity
    {
        readonly string[] permissionGroup =
{
            Manifest.Permission.ReadExternalStorage,
            Manifest.Permission.WriteExternalStorage,
            Manifest.Permission.Camera
        };
        string selected_profile;
        string[] person_list ;
        protected override void OnCreate(Bundle bundle)
        {

            base.OnCreate(bundle);
            string person_list_str = Intent.GetStringExtra("person_list" ?? "Not recv");
            person_list_str = person_list_str.Replace("\0", " ");
            person_list_str = person_list_str.Trim();
            //person_list = person_list_str.Split(' ');
            person_list = new string[] {"Magda", "Leszek", "Joana", "Zenon", "Kasia" };
            var person_list_temp = person_list.ToList();
            person_list_temp.Add("Dodaj nowy");
            person_list = person_list_temp.ToArray();
            
            ListAdapter = new ArrayAdapter<string>(this, Resource.Layout.PersonListPage, person_list);
            ListView.TextFilterEnabled = true;
          
            RequestPermissions(permissionGroup, 0);
            ListView.ItemClick += delegate (object sender, AdapterView.ItemClickEventArgs args)
            {
                //Toast.MakeText(Application, ((TextView)args.View).Text, ToastLength.Short).Show();
                selected_profile = ((TextView)args.View).Text;
                TakePhoto();
                
            };
        }

        async void TakePhoto()
        {
            await CrossMedia.Current.Initialize();
            var file = await CrossMedia.Current.TakePhotoAsync(new Plugin.Media.Abstractions.StoreCameraMediaOptions
            {
                PhotoSize = Plugin.Media.Abstractions.PhotoSize.MaxWidthHeight,
                CompressionQuality = 40,
                Name = "image.png",
                Directory = "sample"
            });

            if (file == null)
            {
                return;
            }

            byte[] imageArray = System.IO.File.ReadAllBytes(file.Path);
            Android.Graphics.Bitmap bitmap = BitmapFactory.DecodeByteArray(imageArray, 0, imageArray.Length);
            byte[] comand = System.Text.ASCIIEncoding.ASCII.GetBytes("Set_picture");
            byte[] name = System.Text.ASCIIEncoding.ASCII.GetBytes(selected_profile);
            byte[] Message = new byte[100 + 100 + imageArray.Length];
            System.Array.Copy(comand, 0, Message, 0, comand.Length);
            System.Array.Copy(name, 0, Message, 100, name.Length);
            System.Array.Copy(imageArray, 0, Message, 200,imageArray.Length);
            await MainActivity.cl.SendAsync
                (Message, WebSocketMessageType.Binary, true, System.Threading.CancellationToken.None);


            // Set our view from the "main" layout resource
            Android.Graphics.Drawables.Drawable v = Android.Graphics.Drawables.Drawable.CreateFromPath("@drawable/image.png");
            Toast.MakeText(this, "The photo has been saved to the profile: Joana", ToastLength.Short).Show();

        }

        public override void OnRequestPermissionsResult(int requestCode, string[] permissions, Android.Content.PM.Permission[] grantResults)
        {
            Xamarin.Essentials.Platform.OnRequestPermissionsResult(requestCode, permissions, grantResults);

            base.OnRequestPermissionsResult(requestCode, permissions, grantResults);
        }

        public interface ICustomNotification
        {
            void send(string title, string message);
        }

    }
}