using Android.App;
using Android.Content;
using Android.OS;
using Android.Runtime;
using Android.Views;
using Android.Widget;
using eWizjer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Wizjer
{
    [Activity(Label = "LoginActivity", MainLauncher = true)]
    public class LoginActivity : Activity
    {      
        protected override void OnCreate(Bundle savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            // Create your application here
            SetContentView(Resource.Layout.LoginPage);

            EditText NameLoginEditText = FindViewById<EditText>(Resource.Id.NameLoginEditText);


            EditText PasswordLoginEditText = FindViewById<EditText>(Resource.Id.PasswordLoginEditText);

        
            Button button = FindViewById<Button>(Resource.Id.LoginButton);
            button.Click +=  delegate
            {
                
                string nameLoginText = NameLoginEditText.Text.ToString(); 
                string passwordLoginText = PasswordLoginEditText.Text.ToString();
                if (nameLoginText == "Admin" && passwordLoginText=="Admin")
                {
                    
          
                    
                StartActivity(typeof(MainActivity));

                    
                }
                else
                {
                    
                }

            };

           
            


        }

       
    }
}