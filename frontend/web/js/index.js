let form_data = new FormData()

let ref_name = ''

function file_upload(input_name) {
    input_el = $('#'+input_name)
    var file_name;
    if(input_el[0].files[0]) {
        file_name = input_el[0].files[0].name;
    }
    else {
        propmt('Something bad happened to input')
    }

    if(!file_name.length)
        return;

    let lbl = input_el.prev()
    lbl.text(file_name);
    if(input_name == 'ref-img') {
        ref_name = file_name
    }

    //btn.text("Выбрать");
    //console.log('OK1');

    var image_file = input_el[0].files[0];

    //var formData = new FormData();
    form_data.append(input_name, image_file);
}


$(function() {
    $('#img-cont').hide();
    $('.lds-grid').hide();

    $('button.file_upload').click(() => {
        $('.lds-grid').show();

        $.ajax({
          url: `http://127.0.0.1:4000/upload`,
          type: 'POST',
          data: form_data,
          async: true,
          cache: false,
          contentType: false,
          processData: false,
          success: function (status) {
            console.log(status);
            $('.lds-grid').hide();
            status = JSON.parse(status)
                let res_name = status['result']
                $('#ref-img-img').attr('src', 'static/images/'+ref_name)
                $('#target-img-img').attr('src', 'static/images/'+res_name)
                $('#result-img').attr('src', 'static/results/'+res_name)
                //$('#img2').attr('src', 'static/'+randname)
                //$('#imlabel1').text(fsize)
                //$//('#imlabel2').text(fsize)
                //$('#img-cont').show();

          }
        });
    })

    // Crutches for the :focus style:
    /*inp.focus(function(){
        wrapper.addClass( "focus" );
    }).blur(function(){
        wrapper.removeClass( "focus" );
    });*/

    $('#ref-img').change(() => {file_upload('ref-img')})
    $('#target-img').change(() => {file_upload('target-img')})
});
