#!/usr/bin/perl -w


# * NiuTrans.NMT v1.0                   * #
# * Author: Qiang Li,                   * #
# * Email : liqiangneu@gmail.com        * #
# * Date  : 10/30/2016                  * #
# * Time  : 15:31                       * #
# * NiuTrans.NMT-bilingual-max-length   * #


use strict;
#use Encode;
#use utf8;

my $logo = "Set max length for source & target sentences\n";

print STDERR $logo;

my %param;


GetParameter( @ARGV );
SetMaxLength();



sub SetMaxLength
{

  print STDERR "MAX_SRC=$param{'-maxlenchn'} MAX_TGT=$param{'-maxleneng'}\n";

  open(SRCINFILE, "<", $param{"-chn"}) or die "Error: can not read file $param{ '-chn' }.\n";
  open(TGTINFILE, "<", $param{"-eng"}) or die "Error: can not read file $param{ '-eng' }.\n";
  open(SRCOUTFILE, ">", $param{"-outchn"}) or die "Error: can not write file $param{'-outchn'}.\n";
  open(TGTOUTFILE, ">", $param{"-outeng"}) or die "Error: can not write file $param{'-outeng'}.\n";
  if ($param{"-outfilter"} eq 1)
  {
    open(SRCOUTFILEFIL, ">", $param{"-outchn"}.".filter") or die "Error: can not write file $param{'-outchn'}.filter.\n";
    open(TGTOUTFILEFIL, ">", $param{"-outeng"}.".filter") or die "Error: can not write file $param{'-outeng'}.filter.\n";
  }
  
  my $line_num = 0;
  my $removed_num = 0;
  my $reserved_num = 0;
  while (my $line_src = <SRCINFILE>)
  {
    my $line_tgt = <TGTINFILE>;
    $line_src =~ s/[\r\n]//g;
    $line_tgt =~ s/[\r\n]//g;

    my $line_src_bak = $line_src;
    my $line_tgt_bak = $line_tgt;
    
    ++$line_num;

    my @src_chars_clean = split /\s+/, $line_src;

    my @tgt_words = split /\s+/, $line_tgt;
    
    if ((scalar(@src_chars_clean) <= $param{"-maxlenchn"}) and (scalar(@tgt_words) <= $param{"-maxleneng"}))
    {
  
      ++$reserved_num;
      print SRCOUTFILE $line_src_bak."\n";
      print TGTOUTFILE $line_tgt_bak."\n";
    }
    else
    {
      ++$removed_num;
      if ($param{"-outfilter"}) 
      {
        print SRCOUTFILEFIL scalar(@src_chars_clean)."\t".$line_src_bak."\n";
        print TGTOUTFILEFIL scalar(@tgt_words)."\t".$line_tgt_bak."\n";
      }
    }
    
    if ($line_num % 10000 == 0)
    {
      print STDERR "\r  $line_num remove=$removed_num reserve=$reserved_num";
    }
  }
  print STDERR "\r  $line_num remove=$removed_num reserve=$reserved_num\n";

  close(SRCINFILE);
  close(TGTINFILE);
  close(SRCOUTFILE);
  close(TGTOUTFILE);
  if ($param{"-outfilter"} eq 1)
  {
    close(SRCOUTFILEFIL);
    close(TGTOUTFILEFIL);
  }
}


sub GetParameter
{
  if( ( scalar( @_ ) < 4 ) || ( scalar( @_ ) % 2 != 0 ) )
  {
    print STDERR "[USAGE]\n".
                 "         NiuTrans.NMT-bilingual-max-length.pl      [OPTIONS]\n".
                 "[OPTION]\n".
                 "          -chn    :  Input chinese file.\n".
                 "          -eng    :  Input english file.\n".
                 "          -outchn :  Output filtered chinese file.\n".
                 "          -outeng :  Output filtered english file.\n".
                 "       -maxlenchn :  Max length for chn.\n".
                 "                       Default is 20 words\n".
                 "       -maxleneng :  Max length for eng.\n".
                 "                       Default is 20 words\n".
                 "       -outfilter :  Out filtered sents mode.\n".
                 "                       Default is false\n".
                 "[EXAMPLE]\n".
                 "     perl NiuTrans.NMT-bilingual-max-length.pl  [-chn    FILE]\n".
                 "                                                [-eng    FILE]\n".
                 "                                                [-outchn FILE]\n".
                 "                                                [-outeng FILE]\n";
    exit( 0 );
  }
          
  my $pos;
  for( $pos = 0; $pos < scalar( @_ ); ++$pos )
  {
    my $key = $ARGV[ $pos ];
    ++$pos;
    my $value = $ARGV[ $pos ];
    $param{ $key } = $value;
  }
          
  if( !exists $param{ "-chn" } )
  {
    print STDERR "Error: please assign \"-chn\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-eng" } )
  {
    print STDERR "Error: please assign \"-eng\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-outchn" } )
  {
    print STDERR "Error: please assign \"-outchn\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-outeng" } )
  {
    print STDERR "Error: please assign \"-outeng\"!\n";
    exit( 1 );
  }

  if( !exists $param{ "-maxlenchn" } )
  {
    $param{ "-maxlenchn" } = 20;
  }

  if( !exists $param{ "-maxleneng" } )
  {
    $param{ "-maxleneng" } = 20;
  }  
  
  if (!exists $param{ "-outfilter" })
  {
    $param{ "-outfilter" } = 0;
  }

}
